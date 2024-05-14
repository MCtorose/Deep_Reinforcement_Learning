#include "flyBall.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

static flyBall flyBall_Obj;

void rt_OneStep(void);

void rt_OneStep(void)
{
    static boolean_T OverrunFlag{false};

    if (OverrunFlag)
    {
        rtmSetErrorStatus(flyBall_Obj.getRTM(), "Overrun");
        return;
    }

    OverrunFlag = true;
    flyBall_Obj.step();
    OverrunFlag = false;
}

int_T main(int_T argc, const char *argv[])
{
    (void)(argc);
    (void)(argv);
    flyBall_Obj.initialize();
    while ((rtmGetErrorStatus(flyBall_Obj.getRTM()) == (nullptr)) && !rtmGetStopRequested(flyBall_Obj.getRTM()))
    {
        rt_OneStep();
    }

    flyBall_Obj.terminate();
    return 0;
}

void reset()
{
    flyBall_Obj.initialize();
}
void step(real_T control)
{
    flyBall_Obj.set_control(control);
    rt_OneStep();
}
// 获取状态量 state=[t,y,vy,ay,target_t,target_y,target_vy]

py::array_t<double> get_state()
{
    double state[7];
    state[0] = flyBall_Obj.get_t();
    state[1] = flyBall_Obj.get_y();
    state[2] = flyBall_Obj.get_vy();
    state[3] = flyBall_Obj.get_ay();
    state[4] = flyBall_Obj.get_target_t();
    state[5] = flyBall_Obj.get_target_y();
    state[6] = flyBall_Obj.get_target_vy();
    return py::array_t<double>(7, state);
}

py::array_t<double> state()
{
    double state[7];
    state[0] = flyBall_Obj.get_t();
    state[1] = flyBall_Obj.get_y();
    state[2] = flyBall_Obj.get_vy();
    state[3] = flyBall_Obj.get_ay();
    state[4] = flyBall_Obj.get_target_t();
    state[5] = flyBall_Obj.get_target_y();
    state[6] = flyBall_Obj.get_target_vy();
    return py::array_t<double>(7, state);
}

double reward()
{
    return flyBall_Obj.get_reward();
}

bool done()
{
    return ((rtmGetErrorStatus(flyBall_Obj.getRTM()) != (nullptr)) || rtmGetStopRequested(flyBall_Obj.getRTM()));
}

void set_control(double control)
{
    flyBall_Obj.set_control(control);
}

PYBIND11_MODULE(flyBall, m)
{
    m.def("reset", &reset);
    m.def("step", &step);
    m.def("get_real_state", &get_state);
    m.def("state", &state);
    m.def("reward", &reward);
    m.def("done", &done);
    m.def("set_control", &set_control);
}
