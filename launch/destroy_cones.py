import carla

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)

world = client.get_world()
world.wait_for_tick()

actor_list = world.get_actors()

cones = actor_list.filter("static.prop.*cone")

for cone in cones:
    cone.destroy()
