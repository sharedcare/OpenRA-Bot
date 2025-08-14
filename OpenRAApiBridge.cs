#region Copyright & License Information
/*
 * OpenRA Python API Bridge
 * Provides HTTP interface for Python RL agents
 */
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.IO;
using OpenRA.Traits;
using OpenRA.Mods.Common.Traits;
using OpenRA.Mods.Common;

namespace OpenRA.Mods.Common.PythonBridge
{
    [Desc("Provides HTTP API for Python RL agents")]
    [TraitLocation(SystemActors.Player)]
    public class PythonApiBridgeInfo : TraitInfo, IBotInfo
    {
        [Desc("Port for HTTP API server")]
        public readonly int ApiPort = 8081;

        [Desc("Enable real-time state broadcasting")]
        public readonly bool EnableRealTimeUpdates = true;

        public string Type => "python-rl-agent";
        public string Name => "Python RL Agent";

        public override object Create(ActorInitializer init) 
        { 
            return new PythonApiBridge(init.Self, this); 
        }
    }

    public class PythonApiBridge : ITick, IBot, INotifyDamage, IDisposable
    {
        private readonly PythonApiBridgeInfo info;
        private readonly World world;
        private Player player;
        private HttpListener httpListener;
        private readonly Queue<Order> pendingOrders = new();
        private readonly List<HttpListenerContext> longPollingClients = new();
        private GameStateData lastGameState;
        private bool serverStarted;

        public IBotInfo Info => info;
        public Player Player => player;

        public PythonApiBridge(Actor self, PythonApiBridgeInfo info)
        {
            this.info = info;
            this.world = self.World;
            this.serverStarted = false;
        }

        public void Activate(Player p)
        {
            player = p;
            if (!serverStarted)
            {
                try
                {
                    StartHttpServer();
                    serverStarted = true;
                }
                catch (HttpListenerException ex)
                {
                    // If another instance already started the listener (unexpected), ignore to avoid crashing
                    Console.WriteLine($"[PythonApiBridge] HTTP listener error: {ex.Message}");
                }
            }
        }

        public void QueueOrder(Order order)
        {
            pendingOrders.Enqueue(order);
        }

        void ITick.Tick(Actor self)
        {
            // Process pending orders
            while (pendingOrders.Count > 0)
            {
                var order = pendingOrders.Dequeue();
                world.IssueOrder(order);
            }

            // Send game state updates to long-polling clients
            if (info.EnableRealTimeUpdates && longPollingClients.Count > 0)
            {
                var gameState = CollectGameState();
                if (HasGameStateChanged(gameState))
                {
                    var stateJson = JsonSerializer.Serialize(gameState);
                    BroadcastToLongPollingClients(stateJson);
                    lastGameState = gameState;
                }
            }
        }

        void INotifyDamage.Damaged(Actor self, AttackInfo e)
        {
            // Send damage events to long-polling clients
            var damageEvent = new
            {
                type = "damage",
                attacker = e.Attacker?.ActorID,
                defender = self.ActorID,
                damage = e.Damage.Value,
                timestamp = world.WorldTick
            };
            
            var eventJson = JsonSerializer.Serialize(damageEvent);
            BroadcastToLongPollingClients(eventJson);
        }

        private void StartHttpServer()
        {
            httpListener = new HttpListener();
            httpListener.Prefixes.Add($"http://localhost:{info.ApiPort}/");
            httpListener.Start();

            Task.Run(async () =>
            {
                while (httpListener.IsListening)
                {
                    try
                    {
                        var context = await httpListener.GetContextAsync();
                        await HandleHttpRequest(context);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"HTTP Error: {ex.Message}");
                    }
                }
            });
        }

        private async Task HandleHttpRequest(HttpListenerContext context)
        {
            var request = context.Request;
            var response = context.Response;

            try
            {
                string responseText = "";

                switch (request.Url.AbsolutePath)
                {
                    case "/api/gamestate":
                        responseText = JsonSerializer.Serialize(CollectGameState());
                        break;
                    
                    case "/api/gamestate/stream":
                        // Long-polling endpoint for real-time updates
                        if (info.EnableRealTimeUpdates)
                        {
                            lock (longPollingClients)
                            {
                                longPollingClients.Add(context);
                            }
                            return; // Don't close the response yet
                        }
                        else
                        {
                            responseText = JsonSerializer.Serialize(CollectGameState());
                        }
                        break;
                    
                    case "/api/actions":
                        if (request.HttpMethod == "POST")
                        {
                            var body = await new StreamReader(request.InputStream).ReadToEndAsync();
                            var actions = JsonSerializer.Deserialize<ActionRequest[]>(body);
                            ProcessActions(actions);
                            responseText = JsonSerializer.Serialize(new { success = true });
                        }
                        break;

                    case "/api/reset":
                        // Reset game to initial state
                        ResetGame();
                        responseText = JsonSerializer.Serialize(new { success = true });
                        break;

                    default:
                        response.StatusCode = 404;
                        responseText = "Not Found";
                        break;
                }

                var buffer = System.Text.Encoding.UTF8.GetBytes(responseText);
                response.ContentLength64 = buffer.Length;
                response.ContentType = "application/json";
                await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
            }
            catch (Exception ex)
            {
                response.StatusCode = 500;
                var errorBuffer = System.Text.Encoding.UTF8.GetBytes($"{{\"error\": \"{ex.Message}\"}}");
                await response.OutputStream.WriteAsync(errorBuffer, 0, errorBuffer.Length);
            }
            finally
            {
                response.Close();
            }
        }

        private GameStateData CollectGameState()
        {
            if (player == null) return new GameStateData();

            var myUnits = player.World.Actors
                .Where(a => a.Owner == player && !a.IsDead && a.IsInWorld && a.OccupiesSpace != null)
                .Select(a => new ActorData
                {
                    Id = a.ActorID,
                    Type = a.Info.Name,
                    Position = new PositionData { X = a.Location.X, Y = a.Location.Y },
                    Health = a.TraitOrDefault<IHealth>()?.HP ?? 0,
                    MaxHealth = a.TraitOrDefault<IHealth>()?.MaxHP ?? 0,
                    IsIdle = a.IsIdle
                }).ToArray();

            var enemyUnits = player.World.Actors
                .Where(a => a.Owner != player && !a.IsDead && a.IsInWorld 
                           && a.OccupiesSpace != null
                           && player.RelationshipWith(a.Owner) == PlayerRelationship.Enemy)
                .Where(a => player.Shroud.IsVisible(a.Location)) // Only visible enemies
                .Select(a => new ActorData
                {
                    Id = a.ActorID,
                    Type = a.Info.Name,
                    Position = new PositionData { X = a.Location.X, Y = a.Location.Y },
                    Health = a.TraitOrDefault<IHealth>()?.HP ?? 0,
                    MaxHealth = a.TraitOrDefault<IHealth>()?.MaxHP ?? 0
                }).ToArray();

            var allyUnits = player.World.Actors
                .Where(a => a.Owner != player && !a.IsDead && a.IsInWorld
                           && a.OccupiesSpace != null
                           && player.RelationshipWith(a.Owner) == PlayerRelationship.Ally)
                .Where(a => player.Shroud.IsVisible(a.Location)) // Only include visible allies
                .Select(a => new ActorData
                {
                    Id = a.ActorID,
                    Type = a.Info.Name,
                    Position = new PositionData { X = a.Location.X, Y = a.Location.Y },
                    Health = a.TraitOrDefault<IHealth>()?.HP ?? 0,
                    MaxHealth = a.TraitOrDefault<IHealth>()?.MaxHP ?? 0
                }).ToArray();

            var playerResources = player.PlayerActor.Trait<PlayerResources>();
            var powerManager = player.PlayerActor.TraitOrDefault<PowerManager>();

            // Map info enrichment
            var map = world.Map;
            var mapInfo = new MapInfoData
            {
                Tileset = map.Tileset,
                Bounds = new BoundsData
                {
                    X = map.Bounds.X,
                    Y = map.Bounds.Y,
                    Width = map.Bounds.Width,
                    Height = map.Bounds.Height
                },
                ResourceCells = Array.Empty<ResourceCellData>()
            };

            // Collect visible resource cells if resource layer is available
            var resourceLayer = world.WorldActor.TraitOrDefault<IResourceLayer>();
            if (resourceLayer != null)
            {
                var resources = new List<ResourceCellData>();
                foreach (var cell in map.AllCells)
                {
                    if (!resourceLayer.IsVisible(cell))
                        continue;

                    var content = resourceLayer.GetResource(cell);
                    if (content.Type == null || content.Density == 0)
                        continue;

                    resources.Add(new ResourceCellData
                    {
                        X = cell.X,
                        Y = cell.Y,
                        Type = content.Type,
                        Density = content.Density
                    });
                }

                mapInfo.ResourceCells = resources.ToArray();
            }

            return new GameStateData
            {
                Tick = world.WorldTick,
                MyUnits = myUnits,
                EnemyUnits = enemyUnits,
                AllyUnits = allyUnits,
                Resources = new ResourceData
                {
                    Cash = playerResources.Cash,
                    Resources = playerResources.Resources,
                    ResourceCapacity = playerResources.ResourceCapacity
                },
                Power = new PowerData
                {
                    Provided = powerManager?.PowerProvided ?? 0,
                    Drained = powerManager?.PowerDrained ?? 0,
                    State = powerManager?.PowerState.ToString() ?? "Normal"
                },
                MapSize = new PositionData 
                { 
                    X = world.Map.MapSize.Width,
                    Y = world.Map.MapSize.Height
                },
                Map = mapInfo,
                Production = CollectProductionInfo()
            };
        }

        private ProductionOverview CollectProductionInfo()
        {
            var overview = new ProductionOverview
            {
                Queues = Array.Empty<ProductionQueueData>()
            };

            var queues = new List<ProductionQueueData>();
            foreach (var a in world.Actors)
            {
                if (a.Owner != player || a.IsDead || !a.IsInWorld)
                    continue;

                var pqs = a.TraitsImplementing<ProductionQueue>().ToArray();
                if (pqs.Length == 0)
                    continue;

                foreach (var q in pqs)
                {
                    if (!q.Enabled)
                        continue;

                    var current = q.CurrentItem();
                    var items = q.AllQueued().Select(pi => new ProductionQueueItem
                    {
                        Item = pi.Item,
                        Cost = pi.TotalCost,
                        Progress = pi.TotalTime > 0 ? Math.Clamp((int)(((pi.TotalTime - pi.RemainingTime) * 100) / pi.TotalTime), 0, 100) : 0,
                        Paused = pi.Paused,
                        Done = pi.Done
                    }).ToArray();

                    var buildables = q.BuildableItems().Select(ai => new BuildableItem
                    {
                        Name = ai.Name,
                        Cost = q.GetProductionCost(ai)
                    }).ToArray();

                    queues.Add(new ProductionQueueData
                    {
                        ActorId = a.ActorID,
                        Type = q.Info.Type,
                        Group = q.Info.Group,
                        Enabled = q.Enabled,
                        Current = current != null ? new ProductionQueueItem
                        {
                            Item = current.Item,
                            Cost = current.TotalCost,
                            Progress = current.TotalTime > 0 ? Math.Clamp((int)(((current.TotalTime - current.RemainingTime) * 100) / current.TotalTime), 0, 100) : 0,
                            Paused = current.Paused,
                            Done = current.Done
                        } : null,
                        Items = items,
                        Buildable = buildables
                    });
                }
            }

            overview.Queues = queues.ToArray();
            return overview;
        }

        private void ProcessActions(ActionRequest[] actions)
        {
            foreach (var action in actions)
            {
                switch (action.Type)
                {
                    case "move":
                        ProcessMoveAction(action);
                        break;
                    case "attack":
                        ProcessAttackAction(action);
                        break;
                    case "deploy":
                        ProcessDeployAction(action);
                        break;
                    case "build":
                        ProcessBuildAction(action);
                        break;
                    case "produce":
                        ProcessProduceAction(action);
                        break;
                }
            }
        }

        private void ProcessMoveAction(ActionRequest action)
        {
            var actor = world.GetActorById(action.ActorId);
            if (actor != null && actor.Owner == player)
            {
                var targetCell = new CPos(action.TargetX, action.TargetY);
                var order = new Order("Move", actor, Target.FromCell(world, targetCell), false);
                QueueOrder(order);
            }
        }

        private void ProcessAttackAction(ActionRequest action)
        {
            var actor = world.GetActorById(action.ActorId);
            var target = world.GetActorById(action.TargetId);
            
            if (actor != null && target != null && actor.Owner == player)
            {
                var order = new Order("Attack", actor, Target.FromActor(target), false);
                QueueOrder(order);
            }
        }

        private void ProcessBuildAction(ActionRequest action)
        {
            // Implementation for building structures
            var builder = world.GetActorById(action.ActorId);
            if (builder != null && builder.Owner == player)
            {
                var targetCell = new CPos(action.TargetX, action.TargetY);
                // Create appropriate build order
            }
        }

        private void ProcessProduceAction(ActionRequest action)
        {
            var producer = world.GetActorById(action.ActorId);
            if (producer != null && producer.Owner == player)
            {
                var order = Order.StartProduction(producer, action.UnitType, 1, true);
                QueueOrder(order);
            }
        }

        private void ProcessDeployAction(ActionRequest action)
        {
            var actor = world.GetActorById(action.ActorId);
            if (actor != null && actor.Owner == player)
            {
                // Deploy MCV or other units with Transforms/Deploy capability
                var order = new Order("DeployTransform", actor, false);
                QueueOrder(order);
            }
        }

        private void ResetGame()
        {
            // Implementation for resetting game state
            // This might involve reloading the map or resetting player states
            // For now, we'll just clear pending orders
            pendingOrders.Clear();
        }

        private bool HasGameStateChanged(GameStateData newState)
        {
            if (lastGameState == null) return true;
            
            // Simple comparison - you might want to make this more sophisticated
            return lastGameState.Tick != newState.Tick ||
                   lastGameState.MyUnits.Length != newState.MyUnits.Length ||
                   lastGameState.EnemyUnits.Length != newState.EnemyUnits.Length ||
                   lastGameState.Resources.Cash != newState.Resources.Cash;
        }

        private void BroadcastToLongPollingClients(string message)
        {
            var clientsToRemove = new List<HttpListenerContext>();
            
            lock (longPollingClients)
            {
                foreach (var client in longPollingClients)
                {
                    try
                    {
                        var buffer = Encoding.UTF8.GetBytes(message);
                        client.Response.ContentLength64 = buffer.Length;
                        client.Response.ContentType = "application/json";
                        client.Response.OutputStream.Write(buffer, 0, buffer.Length);
                        client.Response.Close();
                        clientsToRemove.Add(client);
                    }
                    catch
                    {
                        // Client disconnected, mark for removal
                        clientsToRemove.Add(client);
                    }
                }
                
                // Remove disconnected clients
                foreach (var client in clientsToRemove)
                {
                    longPollingClients.Remove(client);
                }
            }
        }

        public void Dispose()
        {
            httpListener?.Stop();
            
            // Close all long-polling connections
            lock (longPollingClients)
            {
                foreach (var client in longPollingClients)
                {
                    try
                    {
                        client.Response.Close();
                    }
                    catch { }
                }
                longPollingClients.Clear();
            }
        }
    }

    // Data transfer objects
    public class GameStateData
    {
        public int Tick { get; set; }
        public ActorData[] MyUnits { get; set; } = Array.Empty<ActorData>();
        public ActorData[] EnemyUnits { get; set; } = Array.Empty<ActorData>();
        public ActorData[] AllyUnits { get; set; } = Array.Empty<ActorData>();
        public ResourceData Resources { get; set; } = new();
        public PowerData Power { get; set; } = new();
        public PositionData MapSize { get; set; } = new();
        public MapInfoData Map { get; set; } = new();
        public ProductionOverview Production { get; set; } = new();
    }

    public class ActorData
    {
        public uint Id { get; set; }
        public string Type { get; set; } = "";
        public PositionData Position { get; set; } = new();
        public int Health { get; set; }
        public int MaxHealth { get; set; }
        public bool IsIdle { get; set; }
    }

    public class PositionData
    {
        public int X { get; set; }
        public int Y { get; set; }
    }

    public class ResourceData
    {
        public int Cash { get; set; }
        public int Resources { get; set; }
        public int ResourceCapacity { get; set; }
    }

    public class PowerData
    {
        public int Provided { get; set; }
        public int Drained { get; set; }
        public string State { get; set; } = "Normal";
    }

    public class ActionRequest
    {
        public string Type { get; set; } = "";
        public uint ActorId { get; set; }
        public uint TargetId { get; set; }
        public int TargetX { get; set; }
        public int TargetY { get; set; }
        public string UnitType { get; set; } = "";
        public bool Queued { get; set; } = false;
    }

    public class MapInfoData
    {
        public string Tileset { get; set; } = "";
        public BoundsData Bounds { get; set; } = new();
        public ResourceCellData[] ResourceCells { get; set; } = Array.Empty<ResourceCellData>();
    }

    public class BoundsData
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public class ResourceCellData
    {
        public int X { get; set; }
        public int Y { get; set; }
        public string Type { get; set; } = "";
        public byte Density { get; set; }
    }

    public class ProductionOverview
    {
        public ProductionQueueData[] Queues { get; set; } = Array.Empty<ProductionQueueData>();
    }

    public class ProductionQueueData
    {
        public uint ActorId { get; set; }
        public string Type { get; set; } = "";
        public string Group { get; set; } = null;
        public bool Enabled { get; set; }
        public ProductionQueueItem Current { get; set; }
        public ProductionQueueItem[] Items { get; set; } = Array.Empty<ProductionQueueItem>();
        public BuildableItem[] Buildable { get; set; } = Array.Empty<BuildableItem>();
    }

    public class ProductionQueueItem
    {
        public string Item { get; set; } = "";
        public int Cost { get; set; }
        public int Progress { get; set; }
        public bool Paused { get; set; }
        public bool Done { get; set; }
    }

    public class BuildableItem
    {
        public string Name { get; set; } = "";
        public int Cost { get; set; }
    }
}
