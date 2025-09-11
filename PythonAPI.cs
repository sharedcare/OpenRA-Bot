using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using OpenRA.Network;
using OpenRA.Primitives;
using OpenRA.Traits;

namespace OpenRA
{
	// Minimal DTOs for pythonnet interop
	public sealed class RLTarget
	{
		public string Type; // "None", "Cell", "Actor"
		public int CellBits; // for cell targets (CPos.Bits)
		public uint ActorId; // for actor targets
		public byte SubCell; // optional
	}

	public sealed class RLAction
	{
		public string Order; // e.g., "Move", "Attack", mod-specific
		public uint SubjectActorId;
		public bool Queued;
		public RLTarget Target;
		public string TargetString; // optional
		public int ExtraData; // optional
	}

	public sealed class RLActor
	{
		public uint ActorId;
		public string Type;
		public int OwnerIndex;
		public int CellBits;
		public int CellX;
		public int CellY;
		public int HP;
		public int MaxHP;
		public bool IsDead;
		public string[] AvailableOrders;
	}

	public sealed class RLState
	{
		public int WorldTick;
		public int NetFrame;
		public int LocalFrame;
		public RLActor[] Actors;
	}

	/// <summary>
	/// A small bridge exposing StartLocalGame/Step/GetState/SendActions for pythonnet.
	/// </summary>
	public static class PythonAPI
	{
		static bool initialized;

		static void EnsureInitialized()
		{
			if (initialized)
				return;
			initialized = true;
		}

		// Start a local game (single process, EchoConnection) for a given mod and map.
		// modId: e.g., "ra", "cnc"; mapUid: from Mods' map cache.
		public static void StartLocalGame(string modId, string mapUid)
		{
			EnsureInitialized();

			// Fallback to assembly directory as BinDir if a specific binDir is not provided
			var asmDir = Path.GetDirectoryName(typeof(PythonAPI).Assembly.Location);
			StartLocalGame(modId, mapUid, asmDir);
		}

		// Overload that accepts an explicit binDir to ensure Platform.BinDir resolves correctly
		public static void StartLocalGame(string modId, string mapUid, string binDir)
		{
			EnsureInitialized();

			var desiredBinDir = string.IsNullOrEmpty(binDir) ?
				Path.GetFullPath(Path.GetDirectoryName(typeof(PythonAPI).Assembly.Location) ?? ".") :
				Path.GetFullPath(binDir);
			var engineDir = Directory.GetParent(desiredBinDir)?.FullName ?? desiredBinDir;

			// Hint the runtime base directory used by Platform.BinDir
			// This influences AppDomain.CurrentDomain.BaseDirectory on .NET Core hosts
			try { AppDomain.CurrentDomain.SetData("APP_CONTEXT_BASE_DIRECTORY", desiredBinDir); } catch { }

			// Initialize the engine+mod state if needed using the provided binDir for EngineDir
			if (Game.Mods == null || Game.ModData == null || Game.ModData.Manifest.Id != modId)
			{
				var args = new Arguments($"Game.Mod={modId}", $"Engine.EngineDir={engineDir}");
				typeof(Game).GetMethod("Initialize", BindingFlags.NonPublic | BindingFlags.Static)
					.Invoke(null, [args]);
			}

			// Join local echo connection and set up a minimal lobby
			JoinLocalIfNeeded();

			// Ensure there is a playable agent bound to a slot and the map is selected
			EnsurePlayableAgent(mapUid, null);

			// Prepare and start world
			var preview = Game.ModData.MapCache[mapUid];
			if (preview.Status != MapStatus.Available)
				throw new InvalidOperationException($"Invalid map uid: {mapUid}");

			Game.StartGame(preview.ToMap(), WorldType.Regular);
		}

		// Ensure a playable client is bound to a playable slot for the given map
		public static void EnsurePlayableAgent(string mapUid, string slotId)
		{
			var om = Game.OrderManager;
			if (om == null)
				throw new InvalidOperationException("OrderManager not initialized.");

			var lobby = om.LobbyInfo;
			lobby.GlobalSettings.Map = mapUid;

			var desiredSlot = string.IsNullOrEmpty(slotId) ? "Multi0" : slotId;
			if (!lobby.Slots.ContainsKey(desiredSlot))
				lobby.Slots[desiredSlot] = new Session.Slot { PlayerReference = desiredSlot };

			var localId = Game.LocalClientId;
			var client = lobby.Clients.FirstOrDefault(c => c.Index == localId) ?? lobby.Clients.FirstOrDefault();
			if (client != null)
			{
				client.IsAdmin = true;
				client.State = Session.ClientState.Ready;
				client.Slot = desiredSlot;
				client.Faction ??= "Random";
			}
		}

		static void JoinLocalIfNeeded()
		{
			if (Game.OrderManager == null)
			{
				// Use same logic as Game.JoinLocal(), but it's internal. Recreate essential parts here.
				var om = new OrderManager(new EchoConnection());
				// Refresh static classes before the game starts
				TextNotificationsManager.Clear();
				UnitOrders.Clear();

				Game.OrderManager?.Dispose();
				typeof(Game).GetMethod("JoinInner", BindingFlags.NonPublic | BindingFlags.Static)
					.Invoke(null, [om]);

				// Add a single ready client matching local id
				Game.OrderManager.LobbyInfo.Clients.Add(new Session.Client
				{
					Index = Game.OrderManager.Connection.LocalClientId,
					Name = Game.Settings?.Player?.Name ?? "Python",
					PreferredColor = Game.Settings?.Player?.Color ?? Color.FromArgb(0xFFFF00FFu),
					Color = Game.Settings?.Player?.Color ?? Color.FromArgb(0xFFFF00FFu),
					Faction = "Random",
					SpawnPoint = 0,
					Team = 0,
					State = Session.ClientState.Ready
				});
			}
		}

		// Advance one simulation tick, mirroring InnerLogicTick core (no UI timing)
		public static bool Step()
		{
			var om = Game.OrderManager;
			if (om == null || om.World == null)
				return false;

			// Immediate network/connection maintenance
			om.TickImmediate();

			var world = om.World;
			var willTick = om.TryTick();
			if (willTick)
			{
				// Run unsynced order generator tick and world tick
				Sync.RunUnsynced(world, () => world.OrderGenerator.Tick(world));
				world.Tick();
			}

			return willTick;
		}

		// Convert RLAction list to engine orders and queue them for the next frame
		public static void SendActions(IEnumerable<RLAction> actions)
		{
			var world = Game.OrderManager?.World;
			if (world == null)
				return;

			var orders = new List<Order>();
			foreach (var a in actions ?? [])
			{
				var subject = world.GetActorById(a.SubjectActorId);
				if (subject == null || subject.Disposed)
					continue;

				var target = ConvertTarget(world, a.Target);
				var order = new Order(a.Order ?? string.Empty, subject, target, a.Queued)
				{
					TargetString = a.TargetString,
					ExtraData = unchecked((uint)a.ExtraData)
				};

				orders.Add(order);
			}

			foreach (var o in orders)
				world.IssueOrder(o);
		}

		// Overloads to make pythonnet interop more forgiving
		public static void SendActions(params RLAction[] actions)
		{
			if (actions == null || actions.Length == 0)
				return;
			SendActions((IEnumerable<RLAction>)actions);
		}

		public static void SendActions(List<RLAction> actions)
		{
			if (actions == null || actions.Count == 0)
				return;
			SendActions((IEnumerable<RLAction>)actions);
		}

		public static void SendActions(object actions)
		{
			if (actions == null)
				return;

			if (actions is RLAction single)
			{
				SendActions(new[] { single });
				return;
			}

			if (actions is IEnumerable<RLAction> typedEnumerable)
			{
				SendActions(typedEnumerable);
				return;
			}

			if (actions is IEnumerable enumerable)
			{
				var list = new List<RLAction>();
				foreach (var o in enumerable)
					if (o is RLAction ra)
						list.Add(ra);
				SendActions(list);
			}
		}

		static Target ConvertTarget(World world, RLTarget t)
		{
			if (t == null || string.IsNullOrEmpty(t.Type) || t.Type == "None")
				return Target.Invalid;

			switch (t.Type)
			{
				case "Cell":
					return Target.FromCell(world, new CPos(t.CellBits), (SubCell)t.SubCell);
				case "Actor":
					var a = world.GetActorById(t.ActorId);
					return a != null ? Target.FromActor(a) : Target.Invalid;
				default:
					return Target.Invalid;
			}
		}

		// Snapshot the current world state into simple DTOs
		public static RLState GetState()
		{
			var om = Game.OrderManager;
			var world = om?.World;
			if (world == null)
				return new RLState { WorldTick = 0, NetFrame = 0, LocalFrame = 0, Actors = [] };

			var list = new List<RLActor>();
			var localPlayer = world.LocalPlayer ?? world.Players.FirstOrDefault(p => p.ClientIndex == Game.LocalClientId);
			foreach (var a in world.Actors)
			{
				// Align with PythonApiBridge filters: only include units that are relevant
				if (!a.IsInWorld || a.IsDead || a.OccupiesSpace == null)
					continue;

				var owner = a.Owner;
				var include = false;
				if (localPlayer != null)
				{
					if (owner == localPlayer)
						include = true; // my units
					else
					{
						var rel = localPlayer.RelationshipWith(owner);
						if ((rel == PlayerRelationship.Enemy || rel == PlayerRelationship.Ally) && localPlayer.Shroud.IsVisible(a.Location))
							include = true; // visible enemies or allies
					}
				}
				else
					include = true; // fallback if we can't resolve a local player

				if (!include)
					continue;

				var hpTrait = a.TraitsImplementing<IHealth>().FirstOrDefault();
				var availableOrders = a.TraitsImplementing<IIssueOrder>()
					.Where(Exts.IsTraitEnabled)
					.SelectMany(t => t.Orders)
					.Select(o => o.OrderID)
					.Distinct()
					.ToArray();

				list.Add(new RLActor
				{
					ActorId = a.ActorID,
					Type = a.Info?.Name,
					OwnerIndex = a.Owner?.ClientIndex ?? -1,
					CellBits = a.Location.Bits,
					CellX = a.Location.X,
					CellY = a.Location.Y,
					HP = hpTrait?.HP ?? 0,
					MaxHP = hpTrait?.MaxHP ?? 0,
					IsDead = false,
					AvailableOrders = availableOrders
				});
			}

			return new RLState
			{
				WorldTick = world.WorldTick,
				NetFrame = Game.NetFrameNumber,
				LocalFrame = Game.LocalTick,
				Actors = list.ToArray()
			};
		}

		// Strict feasibility check for a specific (actor, order) against a given target.
		// - orderId: IOrderTargeter.OrderID
		// - targetType: "None" | "Cell" | "Actor"
		// - cellBits / subCell or targetActorId used depending on targetType
		public static bool CheckOrderFeasibility(uint subjectActorId, string orderId, string targetType, int cellBits, byte subCell, uint targetActorId,
			bool forceAttack = false, bool forceQueue = false, bool forceMove = false)
		{
			var world = Game.OrderManager?.World;
			if (world == null)
				return false;

			var subject = world.GetActorById(subjectActorId);
			if (subject == null || subject.Disposed || subject.IsDead)
				return false;

			var orders = subject.TraitsImplementing<IIssueOrder>()
				.Where(Exts.IsTraitEnabled)
				.SelectMany(t => t.Orders)
				.Where(o => o.OrderID == orderId)
				.ToList();
			if (orders.Count == 0)
				return false;

			// Build target
			var target = Target.Invalid;
			if (string.Equals(targetType, "Cell", StringComparison.OrdinalIgnoreCase))
				target = Target.FromCell(world, new CPos(cellBits), (SubCell)subCell);
			else if (string.Equals(targetType, "Actor", StringComparison.OrdinalIgnoreCase))
			{
				var ta = world.GetActorById(targetActorId);
				if (ta != null)
					target = Target.FromActor(ta);
			}

			var modifiers = TargetModifiers.None;
			if (forceAttack) modifiers |= TargetModifiers.ForceAttack;
			if (forceQueue) modifiers |= TargetModifiers.ForceQueue;
			if (forceMove) modifiers |= TargetModifiers.ForceMove;

			string cursor = null;
			foreach (var o in orders)
			{
				var localMods = modifiers;
				if (o.CanTarget(subject, target, ref localMods, ref cursor))
					return true;
			}

			return false;
		}

		// --- Remote server join + lobby helpers for pythonnet ---

		public static bool JoinServer(string modId, string host, int port, string password = "", string binDir = null)
		{
			EnsureInitialized();

			var desiredBinDir = string.IsNullOrEmpty(binDir) ?
				Path.GetFullPath(Path.GetDirectoryName(typeof(PythonAPI).Assembly.Location) ?? ".") :
				Path.GetFullPath(binDir);
			var engineDir = Directory.GetParent(desiredBinDir)?.FullName ?? desiredBinDir;

			try { AppDomain.CurrentDomain.SetData("APP_CONTEXT_BASE_DIRECTORY", desiredBinDir); } catch { }

			if (Game.Mods == null || Game.ModData == null || Game.ModData.Manifest.Id != modId)
			{
				var args = new Arguments($"Game.Mod={modId}", $"Engine.EngineDir={engineDir}");
				typeof(Game).GetMethod("Initialize", BindingFlags.NonPublic | BindingFlags.Static)
					.Invoke(null, [args]);
			}

			var endpoint = new ConnectionTarget(host, port);
			Game.JoinServer(endpoint, password ?? "");
			return true;
		}

		public static void CreateAndStartLocalServer(string modId, string mapUid, string binDir)
		{
			CreateAndStartLocalServer(modId, mapUid, binDir, (string[])null);
		}

		public static void CreateAndStartLocalServer(string modId, string mapUid, string binDir, params string[] setupOrderCommands)
		{
			EnsureInitialized();

			var desiredBinDir = string.IsNullOrEmpty(binDir) ?
				Path.GetFullPath(Path.GetDirectoryName(typeof(PythonAPI).Assembly.Location) ?? ".") :
				Path.GetFullPath(binDir);
			var engineDir = Directory.GetParent(desiredBinDir)?.FullName ?? desiredBinDir;

			try { AppDomain.CurrentDomain.SetData("APP_CONTEXT_BASE_DIRECTORY", desiredBinDir); } catch { }

			if (Game.Mods == null || Game.ModData == null || Game.ModData.Manifest.Id != modId)
			{
				var args = new Arguments($"Game.Mod={modId}", $"Engine.EngineDir={engineDir}");
				typeof(Game).GetMethod("Initialize", BindingFlags.NonPublic | BindingFlags.Static)
					.Invoke(null, new object[] { args });
			}

			var orders = new List<Order>();
			if (setupOrderCommands != null)
				foreach (var s in setupOrderCommands)
					if (!string.IsNullOrWhiteSpace(s))
						orders.Add(Order.Command(s));

			Game.CreateAndStartLocalServer(mapUid, orders);
		}

		public static void CreateAndStartLocalServer(string modId, string mapUid, string binDir, List<string> setupOrderCommands)
		{
			CreateAndStartLocalServer(modId, mapUid, binDir, setupOrderCommands?.ToArray());
		}

		public static string[] GetAvailableSlots()
		{
			var om = Game.OrderManager;
			if (om == null || om.LobbyInfo == null)
				return [];

			lock (om.LobbyInfo)
			{
				return om.LobbyInfo.Slots
					.Where(kv => !kv.Value.Closed && om.LobbyInfo.ClientInSlot(kv.Key) == null)
					.Select(kv => kv.Key)
					.ToArray();
			}
		}

		public static void ClaimSlot(string slotId)
		{
			var om = Game.OrderManager;
			if (om == null || string.IsNullOrEmpty(slotId))
				return;
			om.IssueOrder(Order.Command("slot " + slotId));
		}

		public static void SetReady(bool ready)
		{
			var om = Game.OrderManager;
			if (om == null)
				return;
			var state = ready ? Session.ClientState.Ready : Session.ClientState.Invalid;
			om.IssueOrder(Order.Command("state " + state));
		}

		public static void SetSpectator(bool spectate)
		{
			var om = Game.OrderManager;
			if (om == null)
				return;
			om.IssueOrder(Order.Command("spectate " + (spectate ? "1" : "0")));
		}

		public static bool IsInGame()
		{
			var om = Game.OrderManager;
			return om != null && om.World != null && om.GameStarted;
		}
	}
}
