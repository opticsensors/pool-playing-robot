import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used to locate the .urdf file

# Set the gravity
p.setGravity(0, 0, -9.8)

# Load the ground
planeId = p.loadURDF("plane.urdf")

ballId = p.loadURDF(
    "sphere2.urdf",
    [0, 0, 1],
    p.getQuaternionFromEuler([0, 0, 0]),
)
# Set mass of the ball
p.changeDynamics(ballId, -1, mass=10)

# Make another ball with mass 1
ballId2 = p.loadURDF(
    "sphere2.urdf",
    [0, 0, 1],
    p.getQuaternionFromEuler([0, 0, 0]),
)
p.changeDynamics(ballId2, -1, mass=1)

carId = p.loadURDF("racecar/racecar.urdf", basePosition=[0, 1, 0.2])

# Simulate the physics for a given time
for i in range(10000):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    # print positions of both ground and ball
    # ballPos, ballOrn = p.getBasePositionAndOrientation(ballId)
    # print(ballPos,ballOrn)

# Disconnect
p.disconnect()
