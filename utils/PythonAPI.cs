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
		public RLResourceCell[] Resources;
		public ProductionOverview Production;
		public PlaceableAreaData[] PlaceableAreas;
	}

	public sealed class RLResourceCell
	{
		public int CellX;
		public int CellY;
		public int TypeIndex;
		public int Density;
	}

	public sealed class PositionData
	{
		public int X { get; set; }
		public int Y { get; set; }
	}

	// --- Production DTOs for pythonnet interop ---
	public sealed class ProductionOverview
	{
		public ProductionQueueData[] Queues { get; set; } = [];
	}

	public sealed class ProductionQueueData
	{
		public uint ActorId { get; set; }
		public string Type { get; set; } = "";
		public string Group { get; set; } = null;
		public bool Enabled { get; set; }
		public ProductionQueueItem[] Items { get; set; } = [];
		public BuildableItem[] Buildable { get; set; } = [];
	}

	public sealed class ProductionQueueItem
	{
		public string Item { get; set; } = "";
		public int Cost { get; set; }
		public int Progress { get; set; }
		public bool Paused { get; set; }
		public bool Done { get; set; }
	}

	public sealed class BuildableItem
	{
		public string Name { get; set; } = "";
		public int Cost { get; set; }
	}

	public sealed class PlaceableAreaData
	{
		public uint ActorId { get; set; }
		public string UnitType { get; set; } = "";
		public PositionData[] Cells { get; set; } = [];
	}

	/// <summary>
	/// A small bridge exposing StartLocalGame/Step/GetState/SendActions for pythonnet.
	/// </summary>
	public static class PythonAPI
	{
		static bool initialized;
		static int networkConnectTimeoutMs = 10000;

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

		// Start local game with option to add a built-in bot opponent and set lobby options before the world starts
		public static void StartLocalGame(string modId, string mapUid, string binDir,
			bool addBotOpponent, string botType = null, string botSlotId = null,
			bool? explored = null, bool? fog = null)
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

			JoinLocalIfNeeded();
			EnsurePlayableAgent(mapUid, null);

			if (addBotOpponent)
				TryAddLocalBotOpponent(botType, botSlotId);

			// Apply lobby options directly (update GlobalSettings) and also via order for parity
			var om = Game.OrderManager;
			if (om != null)
			{
				if (explored.HasValue)
				{
					SetLobbyBooleanOption("explored", explored.Value);
					om.IssueOrder(Order.Command($"option explored {(explored.Value ? "True" : "False")}"));
				}

				if (fog.HasValue)
				{
					SetLobbyBooleanOption("fog", fog.Value);
					om.IssueOrder(Order.Command($"option fog {(fog.Value ? "True" : "False")}"));
				}
			}

			var preview = Game.ModData.MapCache[mapUid];
			if (preview.Status != MapStatus.Available)
				throw new InvalidOperationException($"Invalid map uid: {mapUid}");

			Game.StartGame(preview.ToMap(), WorldType.Regular);
		}

		static void TryAddLocalBotOpponent(string botType, string botSlotId)
		{
			var om = Game.OrderManager;
			if (om == null)
				return;

			var lobby = om.LobbyInfo;
			var localId = Game.LocalClientId;
			var controller = lobby.Clients.FirstOrDefault(c => c.Index == localId) ?? lobby.Clients.FirstOrDefault();
			if (controller == null)
				return;

			controller.IsAdmin = true; // ensure Game.IsHost so bot logic enables

			var slotId = string.IsNullOrEmpty(botSlotId) ? "Multi1" : botSlotId;
			if (!lobby.Slots.ContainsKey(slotId))
				lobby.Slots[slotId] = new Session.Slot { PlayerReference = slotId, AllowBots = true };

			var bt = botType;
			if (string.IsNullOrEmpty(bt))
				bt = GetBotTypes().FirstOrDefault();
			if (string.IsNullOrEmpty(bt))
				return;

			var nextIndex = lobby.Clients.Count == 0 ? 1 : lobby.Clients.Max(c => c.Index) + 1;
			lobby.Clients.Add(new Session.Client
			{
				Index = nextIndex,
				Name = "AI",
				PreferredColor = controller.PreferredColor,
				Color = controller.Color,
				Faction = "Random",
				SpawnPoint = 0,
				Team = 0,
				State = Session.ClientState.Ready,
				Slot = slotId,
				Bot = bt,
				BotControllerClientIndex = controller.Index
			});
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

			// Phase 1: receive anything from previous frame
			om.TickImmediate();

			var world = om.World;
			var willTick = om.TryTick();
			if (!willTick)
			{
				// Phase 2: for local echo connections, orders sent during TryTick() are only
				// received on the next TickImmediate() — flush once more, then retry.
				om.TickImmediate();
				willTick = om.TryTick();
			}

			if (willTick)
			{
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

				// Special handling for production orders
				if (string.Equals(a.Order, "StartProduction", StringComparison.OrdinalIgnoreCase))
				{
					if (!string.IsNullOrEmpty(a.TargetString))
					{
						var prodOrder = Order.StartProduction(subject, a.TargetString, 1, true);
						orders.Add(prodOrder);
					}

					continue;
				}

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
				return new RLState { WorldTick = 0, NetFrame = 0, LocalFrame = 0, Actors = [], Resources = [], Production = new ProductionOverview(), PlaceableAreas = [] };

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

			// Build resource snapshot from map binary layer (type index + density)
			var resourceCells = new List<RLResourceCell>();
			var map = world.Map;
			var restrictToVisibility = localPlayer != null;
			foreach (var c in map.AllCells)
			{
				if (restrictToVisibility && !localPlayer.Shroud.IsVisible(c))
					continue;

				var uv = c.ToMPos(map);
				var rt = map.Resources[uv];
				if (rt.Type == 0 || rt.Index == 0)
					continue;

				resourceCells.Add(new RLResourceCell
				{
					CellX = c.X,
					CellY = c.Y,
					TypeIndex = rt.Type,
					Density = rt.Index
				});
			}

			return new RLState
			{
				WorldTick = world.WorldTick,
				NetFrame = Game.NetFrameNumber,
				LocalFrame = Game.LocalTick,
				Actors = list.ToArray(),
				Resources = resourceCells.ToArray(),
				Production = CollectProductionInfo(world, localPlayer),
				PlaceableAreas = CollectPlaceableAreas(world, localPlayer)
			};
		}

		static ProductionOverview CollectProductionInfo(World world, Player player)
		{
			var overview = new ProductionOverview { Queues = [] };
			var queues = new List<ProductionQueueData>();

			// Resolve ProductionQueue trait type via reflection to avoid hard assembly dependency
			var pqType = AppDomain.CurrentDomain
				.GetAssemblies()
				.Select(a => a.GetType("OpenRA.Mods.Common.Traits.ProductionQueue", false))
				.FirstOrDefault(t => t != null);
			if (pqType == null)
			{
				overview.Queues = queues.ToArray();
				return overview;
			}

			var traitsMethod = typeof(Actor).GetMethod("TraitsImplementing", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
			var traitsGeneric = traitsMethod?.MakeGenericMethod(pqType);

			foreach (var a in world.Actors)
			{
				if (player != null && a.Owner != player) continue;
				if (a.IsDead || !a.IsInWorld) continue;

				if (traitsGeneric?.Invoke(a, null) is not IEnumerable enumerable) continue;

				foreach (var q in enumerable)
				{
					var enabledProp = q.GetType().GetProperty("Enabled");
					var enabled = enabledProp != null && (bool)enabledProp.GetValue(q);
					if (!enabled) continue;

					// Items
					var itemsList = new List<ProductionQueueItem>();
					var allQueued = q.GetType().GetMethod("AllQueued");
					if (allQueued?.Invoke(q, null) is IEnumerable queuedEnum)
					{
						foreach (var pi in queuedEnum)
						{
							var item = new ProductionQueueItem();
							var piType = pi.GetType();
							item.Item = piType.GetProperty("Item")?.GetValue(pi) as string ?? "";
							item.Cost = (int)(piType.GetProperty("TotalCost")?.GetValue(pi) ?? 0);
							var totalTime = (int)(piType.GetProperty("TotalTime")?.GetValue(pi) ?? 0);
							var remaining = (int)(piType.GetProperty("RemainingTime")?.GetValue(pi) ?? 0);
							item.Progress = totalTime > 0 ? Math.Clamp((totalTime - remaining) * 100 / totalTime, 0, 100) : 0;
							item.Paused = (bool)(piType.GetProperty("Paused")?.GetValue(pi) ?? false);
							item.Done = (bool)(piType.GetProperty("Done")?.GetValue(pi) ?? false);
							itemsList.Add(item);
						}
					}

					// Buildables
					var buildablesList = new List<BuildableItem>();
					var buildableItems = q.GetType().GetMethod("BuildableItems");
					var getCost = q.GetType().GetMethod("GetProductionCost");
					if (buildableItems?.Invoke(q, null) is IEnumerable buildableEnum)
					{
						foreach (var ai in buildableEnum)
						{
							var name = ai.GetType().GetProperty("Name")?.GetValue(ai) as string ?? "";
							var cost = getCost != null ? (int)getCost.Invoke(q, [ai]) : 0;
							buildablesList.Add(new BuildableItem { Name = name, Cost = cost });
						}
					}

					// Queue metadata
					var infoProp = q.GetType().GetProperty("Info");
					var infoVal = infoProp?.GetValue(q);
					var typeStr = infoVal?.GetType().GetProperty("Type")?.GetValue(infoVal) as string ?? "";
					var groupStr = infoVal?.GetType().GetProperty("Group")?.GetValue(infoVal) as string;

					queues.Add(new ProductionQueueData
					{
						ActorId = a.ActorID,
						Type = typeStr,
						Group = groupStr,
						Enabled = enabled,
						Items = itemsList.ToArray(),
						Buildable = buildablesList.ToArray()
					});
				}
			}

			overview.Queues = queues.ToArray();
			return overview;
		}

		static PlaceableAreaData[] CollectPlaceableAreas(World world, Player player)
		{
			// Not available in core Game assembly; requires Mods.Common helpers.
			// Return empty to keep layout consistent with HTTP bridge.
			return [];
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

		// Remote server join + lobby helpers for pythonnet
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

		// Get current connection state
		public static string GetConnectionState()
		{
			var om = Game.OrderManager;
			if (om?.Connection is NetworkConnection nc)
				return nc.ConnectionState.ToString();
			return "NotConnected";
		}

		// Check if connected to lobby
		public static bool IsConnectedToLobby()
		{
			var om = Game.OrderManager;
			if (om?.Connection is NetworkConnection nc)
			{
				// Consider lobby connected when the transport is connected and we have lobby info that includes ourselves
				if (nc.ConnectionState != ConnectionState.Connected || om.LobbyInfo == null)
					return false;

				return om.LocalClient != null; // validated and added to LobbyInfo
			}

			return false;
		}

		// Get lobby information as a dictionary-like structure
		public static Dictionary<string, object> GetLobbyInfo()
		{
			var om = Game.OrderManager;
			if (om?.LobbyInfo == null)
				return [];

			var lobby = om.LobbyInfo;
			var result = new Dictionary<string, object>
			{
				["ServerName"] = lobby.GlobalSettings.ServerName ?? "",
				["Map"] = lobby.GlobalSettings.Map ?? "",
				["MapStatus"] = lobby.GlobalSettings.MapStatus.ToString(),
				["AllowSpectators"] = lobby.GlobalSettings.AllowSpectators,
				["GameUid"] = lobby.GlobalSettings.GameUid ?? "",
				["Dedicated"] = lobby.GlobalSettings.Dedicated
			};

			// Add clients info
			var clients = new List<Dictionary<string, object>>();
			foreach (var client in lobby.Clients)
			{
				clients.Add(new Dictionary<string, object>
				{
					["Index"] = client.Index,
					["Name"] = client.Name ?? "",
					["State"] = client.State.ToString(),
					["Slot"] = client.Slot ?? "",
					["IsAdmin"] = client.IsAdmin,
					["IsObserver"] = client.IsObserver,
					["IsBot"] = client.IsBot,
					["Team"] = client.Team,
					["Faction"] = client.Faction ?? "",
					["Color"] = client.Color.ToArgb()
				});
			}

			result["Clients"] = clients;

			// Add slots info
			var slots = new Dictionary<string, object>();
			foreach (var slot in lobby.Slots)
			{
				slots[slot.Key] = new Dictionary<string, object>
				{
					["PlayerReference"] = slot.Value.PlayerReference ?? "",
					["Closed"] = slot.Value.Closed,
					["AllowBots"] = slot.Value.AllowBots,
					["LockFaction"] = slot.Value.LockFaction,
					["LockColor"] = slot.Value.LockColor,
					["LockTeam"] = slot.Value.LockTeam,
					["Required"] = slot.Value.Required
				};
			}

			result["Slots"] = slots;

			return result;
		}

		// Wait for connection to be established (with timeout)
		public static bool WaitForConnection(int timeoutMs = 10000)
		{
			var effectiveTimeout = timeoutMs > 0 ? timeoutMs : networkConnectTimeoutMs;
			var startTime = Game.RunTime;
			while (Game.RunTime - startTime < effectiveTimeout)
			{
				// Process network messages
				Game.OrderManager?.TickImmediate();

				if (IsConnectedToLobby())
					return true;

				// Check for connection errors
				var om = Game.OrderManager;
				if (om?.Connection is NetworkConnection nc && nc.ConnectionState == ConnectionState.NotConnected)
				{
					if (!string.IsNullOrEmpty(nc.ErrorMessage))
						throw new InvalidOperationException($"Connection failed: {nc.ErrorMessage}");
					return false;
				}

				// Small delay to avoid busy waiting
				System.Threading.Thread.Sleep(50);
			}

			return false;
		}

		public static void SetNetworkConnectTimeout(int milliseconds)
		{
			if (milliseconds < 100)
				milliseconds = 100;
			networkConnectTimeoutMs = milliseconds;
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
					.Invoke(null, [args]);
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
			om.IssueOrder(Order.Command($"state {state}"));
		}

		public static string[] GetBotTypes()
		{
			var om = Game.OrderManager;
			if (om == null || Game.ModData == null)
				return [];

			var uid = om.LobbyInfo?.GlobalSettings?.Map;
			if (string.IsNullOrEmpty(uid))
				return [];

			var preview = Game.ModData.MapCache[uid];
			var types = preview.PlayerActorInfo.TraitInfos<IBotInfo>().Select(t => t.Type).Where(t => t != null).Distinct().ToArray();
			return types;
		}

		public static bool AddBotToSlot(string slotId, string botType = null)
		{
			var om = Game.OrderManager;
			if (om == null || string.IsNullOrEmpty(slotId))
				return false;

			var lobby = om.LobbyInfo;
			var controller = lobby.Clients.FirstOrDefault(c => c.IsAdmin) ?? lobby.Clients.FirstOrDefault();
			if (controller == null)
				return false;

			var bt = botType;
			if (string.IsNullOrEmpty(bt))
				bt = GetBotTypes().FirstOrDefault();
			if (string.IsNullOrEmpty(bt))
				return false;

			om.IssueOrder(Order.Command($"slot_bot {slotId} {controller.Index} {bt}"));
			return true;
		}

		public static bool AddBotToFreeSlot(string botType = null)
		{
			var om = Game.OrderManager;
			if (om == null)
				return false;

			var lobby = om.LobbyInfo;
			foreach (var kv in lobby.Slots)
			{
				var key = kv.Key;
				var slot = kv.Value;
				if (slot.Closed)
					continue;
				if (!slot.AllowBots)
					continue;
				var c = lobby.ClientInSlot(key);
				if (c == null || c.Bot != null)
					return AddBotToSlot(key, botType);
			}

			return false;
		}

		public static void RemoveAllBots()
		{
			var om = Game.OrderManager;
			if (om == null)
				return;
			var lobby = om.LobbyInfo;
			foreach (var kv in lobby.Slots)
			{
				var c = lobby.ClientInSlot(kv.Key);
				if (c != null && c.Bot != null)
					om.IssueOrder(Order.Command("slot_open " + kv.Value.PlayerReference));
			}
		}

		// Directly set a lobby boolean option value in GlobalSettings
		public static void SetLobbyBooleanOption(string id, bool value)
		{
			var om = Game.OrderManager;
			if (om == null || om.LobbyInfo == null)
				return;

			var gs = om.LobbyInfo.GlobalSettings;
			if (!gs.LobbyOptions.TryGetValue(id, out var state))
			{
				state = new Session.LobbyOptionState();
				gs.LobbyOptions[id] = state;
			}

			state.Value = value ? "True" : "False";
		}

		public static void StartGameFromLobby()
		{
			var om = Game.OrderManager;
			if (om == null)
				return;
			om.IssueOrder(Order.Command("startgame"));
		}

		// Inform the server that we have the currently selected map by sending state=NotReady
		// This mirrors the UI lobby logic that acknowledges map availability
		public static bool TryAcknowledgeMap()
		{
			var om = Game.OrderManager;
			if (om == null || om.LobbyInfo == null)
				return false;

			var uid = om.LobbyInfo.GlobalSettings.Map;
			if (string.IsNullOrEmpty(uid))
				return false;

			var preview = Game.ModData?.MapCache?[uid];
			if (preview == null)
				return false;

			if (preview.Status == MapStatus.Available)
			{
				om.IssueOrder(Order.Command($"state {Session.ClientState.NotReady}"));
				return true;
			}

			return false;
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
