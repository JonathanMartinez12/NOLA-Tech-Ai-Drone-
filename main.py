#!/usr/bin/env python3
"""
AI Grand Prix - Autonomous Drone Racing
Main Entry Point

This is the main entry point for the drone racing autonomy stack.
For Phase 1, it demonstrates basic waypoint navigation.

Usage:
    # Basic waypoint mission
    python main.py

    # With custom connection
    python main.py --connection udp://:14540

    # Quick test mode (smaller pattern)
    python main.py --test

Requirements:
    1. PX4 SITL running: cd ~/PX4-Autopilot && make px4_sitl gz_x500
    2. Gazebo Harmonic with default world
    3. Python dependencies installed: pip install -r requirements.txt

Author: AI Grand Prix Competitor
Competition: Anduril AI Grand Prix 2026
"""

import argparse
import asyncio
import signal
import sys
from typing import List, Tuple

from mavsdk import System

from config.settings import get_settings
from control.flight_controller import FlightController
from state_estimation.drone_state import DroneState
from utils.logger import setup_logger, get_logger
from utils.telemetry_recorder import TelemetryRecorder

# Initialize logger
logger = get_logger(__name__)


async def connect_drone(connection_string: str) -> System:
    """
    Connect to the drone via MAVSDK.

    Args:
        connection_string: MAVSDK connection string (e.g., "udp://:14540")

    Returns:
        Connected MAVSDK System instance

    Raises:
        ConnectionError: If connection fails
    """
    logger.info("Connecting to drone", address=connection_string)

    drone = System()
    await drone.connect(system_address=connection_string)

    # Wait for connection
    logger.info("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            logger.info("Drone connected!")
            break

    # Get system info
    async for info in drone.info.get_identification():
        logger.info(
            "Connected to drone",
            hardware_uid=info.hardware_uid,
        )
        break

    return drone


async def wait_for_gps(drone: System, timeout: float = 60.0) -> bool:
    """
    Wait for GPS fix before flight.

    Args:
        drone: MAVSDK System instance
        timeout: Maximum wait time in seconds

    Returns:
        True if GPS fix obtained
    """
    logger.info("Waiting for GPS fix...")
    start_time = asyncio.get_event_loop().time()

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            logger.info("GPS fix acquired, home position set")
            return True

        if asyncio.get_event_loop().time() - start_time > timeout:
            logger.warning("GPS timeout, proceeding anyway (simulation)")
            return True  # In simulation, continue anyway

        await asyncio.sleep(0.5)

    return False


async def run_waypoint_mission(
    controller: FlightController,
    state: DroneState,
    waypoints: List[Tuple[float, float, float]],
    speed: float = 2.0,
    recorder: TelemetryRecorder = None,
) -> bool:
    """
    Execute a waypoint mission.

    This is the Phase 1 core functionality - demonstrating reliable
    waypoint navigation with smooth velocity control.

    Args:
        controller: FlightController instance
        state: DroneState for telemetry
        waypoints: List of (N, E, D) positions in NED frame
        speed: Flight speed in m/s
        recorder: Optional telemetry recorder

    Returns:
        True if mission completed successfully
    """
    logger.info(
        "Starting waypoint mission",
        num_waypoints=len(waypoints),
        speed=speed
    )

    # Start recording if enabled
    if recorder:
        recorder.start()

    try:
        # Arm the drone
        logger.info("Arming drone...")
        if not await controller.arm():
            logger.error("Failed to arm drone")
            return False

        # Takeoff
        takeoff_alt = 5.0
        logger.info("Taking off", altitude=takeoff_alt)
        if not await controller.takeoff(altitude=takeoff_alt):
            logger.error("Takeoff failed")
            return False

        logger.info("Takeoff complete, starting waypoint sequence")

        # Fly to each waypoint
        for i, waypoint in enumerate(waypoints):
            logger.info(
                f"Flying to waypoint {i + 1}/{len(waypoints)}",
                target=waypoint,
                current=state.position_ned,
            )

            if not await controller.goto(waypoint, speed=speed):
                logger.error(f"Failed to reach waypoint {i + 1}")
                return False

            logger.info(f"Reached waypoint {i + 1}")

            # Record state if enabled
            if recorder:
                recorder.record(
                    position=state.position_ned,
                    velocity=state.velocity_ned,
                    attitude=(
                        state.attitude_euler[0] * 0.0174533,  # deg to rad
                        state.attitude_euler[1] * 0.0174533,
                        state.attitude_euler[2] * 0.0174533,
                    ),
                    armed=state.is_armed,
                    flight_mode=state.flight_mode_str,
                )

            # Brief pause at waypoint
            await asyncio.sleep(0.5)

        # Return to home
        logger.info("Mission complete, returning to home")
        if not await controller.return_to_home(speed=speed):
            logger.warning("Failed to return to home position")

        # Land
        logger.info("Landing...")
        if not await controller.land():
            logger.error("Landing failed")
            return False

        logger.info("Mission completed successfully!")
        return True

    except Exception as e:
        logger.error("Mission failed with exception", error=str(e))
        # Emergency stop on error
        await controller.emergency_stop()
        return False

    finally:
        # Stop recording and save
        if recorder:
            recorder.stop()
            filepath = recorder.save()
            stats = recorder.get_statistics()
            logger.info("Flight statistics", **stats)


async def main(args: argparse.Namespace) -> int:
    """
    Main entry point for the drone racing system.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 = success)
    """
    # Setup logging
    setup_logger(log_level="DEBUG" if args.verbose else "INFO")
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("AI Grand Prix - Autonomous Drone Racing")
    logger.info("Phase 1: Waypoint Navigation Demo")
    logger.info("=" * 60)

    # Connect to drone
    try:
        drone = await connect_drone(args.connection)
    except Exception as e:
        logger.error("Failed to connect to drone", error=str(e))
        return 1

    # Initialize state tracking
    state = DroneState(drone)
    await state.start_tracking()

    # Wait for telemetry
    if not await state.wait_for_connection(timeout=30.0):
        logger.error("Failed to receive telemetry")
        return 1

    # Wait for GPS (or timeout in simulation)
    await wait_for_gps(drone, timeout=30.0)

    # Initialize controller
    controller = FlightController(drone, state)

    # Initialize telemetry recorder
    recorder = TelemetryRecorder() if settings.logging.record_telemetry else None

    # Define waypoints based on mode
    if args.test:
        # Small test pattern
        waypoints = [
            (5, 0, -5),    # 5m North
            (5, 5, -5),    # 5m North, 5m East
            (0, 5, -5),    # 5m East
        ]
        logger.info("Test mode: Small triangle pattern")
    else:
        # Standard demo pattern - larger triangle
        waypoints = [
            (10, 0, -5),   # 10m North
            (10, 10, -5),  # 10m North, 10m East
            (0, 10, -5),   # 10m East
        ]
        logger.info("Standard mode: Large triangle pattern")

    logger.info("Waypoints", waypoints=waypoints)

    # Handle shutdown gracefully
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.warning("Shutdown signal received")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run mission
    try:
        mission_task = asyncio.create_task(
            run_waypoint_mission(
                controller=controller,
                state=state,
                waypoints=waypoints,
                speed=args.speed,
                recorder=recorder,
            )
        )

        # Wait for mission or shutdown
        done, pending = await asyncio.wait(
            [mission_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown_event.is_set():
            logger.warning("Mission interrupted by shutdown signal")
            mission_task.cancel()
            # Emergency land
            await controller.land()
            return 1

        # Get mission result
        success = mission_task.result()
        return 0 if success else 1

    except asyncio.CancelledError:
        logger.warning("Mission cancelled")
        return 1

    finally:
        # Cleanup
        await state.stop_tracking()
        logger.info("Shutdown complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Grand Prix - Autonomous Drone Racing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run standard waypoint mission
    python main.py

    # Run small test pattern
    python main.py --test

    # Custom connection and speed
    python main.py --connection udp://:14540 --speed 3.0

    # Verbose logging
    python main.py --verbose
        """,
    )

    parser.add_argument(
        "--connection", "-c",
        type=str,
        default="udp://:14540",
        help="MAVSDK connection string (default: udp://:14540)",
    )

    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=2.0,
        help="Flight speed in m/s (default: 2.0)",
    )

    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run small test pattern instead of full demo",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
