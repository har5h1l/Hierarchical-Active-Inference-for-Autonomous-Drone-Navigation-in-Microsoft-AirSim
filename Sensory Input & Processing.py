import airsim
import time

print("Creating client...")
client = airsim.MultirotorClient()
print("Client created, waiting 5 seconds before confirming connection...")
time.sleep(5)  # Give simulator time to fully initialize
print("Attempting to confirm connection...")
client.confirmConnection()
print("Connection confirmed!")

# Enable API control and arm the drone
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()

# Wait a bit
time.sleep(2)

# Fly forward
print("Flying forward...")
client.moveByVelocityAsync(5, 0, 0, 5).join()

# Land
print("Landing...")
client.landAsync().join()

# Release control
client.armDisarm(False)
client.enableApiControl(False)


