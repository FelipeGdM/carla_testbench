import carla

# center = {"x": -15.15, "y": -69.71}
# center = {"x": 92.1, "y": -28.0}
# center = {"x": 392.5, "y": -105.4}
center = {"x": 189.2, "y": 273.0}
width = 2.3
length = 5.8


cone_locations = [
    carla.Location(x=center["x"] + width / 2, y=center["y"] + length / 2, z=0.2),
    carla.Location(x=center["x"] - width / 2, y=center["y"] + length / 2, z=0.2),
    carla.Location(x=center["x"] + width / 2, y=center["y"] - length / 2, z=0.2),
    carla.Location(x=center["x"] - width / 2, y=center["y"] - length / 2, z=0.2),
    # carla.Location(x=center["x"], y=center["y"], z=2),
]

print(f"Cone Locations: {cone_locations}")

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)

# world = client.get_world()
world = client.get_world()
world.wait_for_tick()

blueprint_library = world.get_blueprint_library()
cone_blueprint = blueprint_library.filter("static.prop.constructioncone")[0]
# cone_blueprint = blueprint_library.filter("static.prop.trafficcone01")[0]
print(f"Cone Blueprint: {cone_blueprint}")

actor_list = world.get_actors()

vehicle_location = actor_list.filter("vehicle.tesla.model3")[0].get_location()

print(f"Vehicle Location: {vehicle_location}")


def spawn_cone(location):
    cone_transform = carla.Transform(
        location, carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )
    # world.wait_for_tick()
    cone = world.spawn_actor(cone_blueprint, cone_transform)
    # world.wait_for_tick()
    print(f"Cone Object: {cone}")
    print(f"Cone Location: {cone.get_location()}")
    return cone


cones = []
for i in range(len(cone_locations)):
    cones.append(spawn_cone(cone_locations[i]))


# print("Actor list")
# print(actor_list.filter("static.prop.trafficcone01"))
# print(actor_list.filter("vehicle.tesla.model3")[0].get_location())
# print(world.get_map().get_spawn_points())
