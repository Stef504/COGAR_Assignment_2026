import argparse
import sys
import roslibpy
from baxter_interface.robot_enable import RobotEnable
from baxter_interface.robust_controller import RobustController

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--limb', required=True, choices=['left', 'right'])
    args = parser.parse_args()

    # WebSocket Connection
    client = roslibpy.Ros(host='130.251.13.31', port=9090)
    client.run()

    print("Enabling Robot...")
    rs = RobotEnable(client)
    rs.enable()

    print(f"Calibrating {args.limb} arm...")
    cat = RobustController(
        client, 
        f'/robustcontroller/{args.limb}/CalibrateArm',
        {'isEnabled': True, 'uid': 'sdk'},
        {'isEnabled': False, 'uid': 'sdk'}
    )
    cat.run()

    rs.disable()
    client.terminate()
    print("Done!")

if __name__ == '__main__':
    main()