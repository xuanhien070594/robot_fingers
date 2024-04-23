/**
 * \file
 * \brief The hardware wrapper of the real Finger robot.
 * \author Manuel Wuthrich
 * \date 2018
 * \copyright Copyright (c) 2019, New York University and Max Planck
 *            Gesellschaft.
 */

#pragma once

#include "n_finger_driver.hpp"

namespace robot_fingers
{
class RealFingerDriver : public NFingerDriver<1>
{
public:
    RealFingerDriver(const Config &config)
        : RealFingerDriver(create_motor_boards(config.can_ports), config)
    {
    }

private:
// suppress warning about designated initializers (e.g. `.torque_constant_NmpA`)
// only being available with C++20 (we will get there eventually so just ignore
// the warning until then).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    RealFingerDriver(const MotorBoards &motor_boards, const Config &config)
        : NFingerDriver<1>(motor_boards,
                           create_motors(motor_boards),
                           {
                               // MotorParameters
                               .torque_constant_NmpA = 0.02,
                               .gear_ratio = 9.0,
                           },
                           config)
    {
    }
#pragma GCC diagnostic pop

    static Motors create_motors(const MotorBoards &motor_boards)
    {
        // set up motors
        Motors motors;
//        motors[0] = std::make_shared<blmc_drivers::Motor>(motor_boards[1], 0);
//        motors[1] = std::make_shared<blmc_drivers::Motor>(motor_boards[0], 0);
//        motors[2] = std::make_shared<blmc_drivers::Motor>(motor_boards[0], 1);

        motors[0] = std::make_shared<blmc_drivers::SafeMotor>(motor_boards[1], 0, 1.0, 1000, 0.5);
        motors[1] = std::make_shared<blmc_drivers::SafeMotor>(motor_boards[0], 0, 1.0, 1000, 0.5);
        motors[2] = std::make_shared<blmc_drivers::SafeMotor>(motor_boards[0], 1, 1.0, 1000, 0.5);
        return motors;
    }
};

}  // namespace robot_fingers
