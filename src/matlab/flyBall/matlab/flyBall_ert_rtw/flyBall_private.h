#ifndef RTW_HEADER_flyBall_private_h_
#define RTW_HEADER_flyBall_private_h_
#include "rtwtypes.h"
#include "flyBall_types.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"

#ifndef rtmIsMajorTimeStep
#define rtmIsMajorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MAJOR_TIME_STEP)
#endif

#ifndef rtmIsMinorTimeStep
#define rtmIsMinorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MINOR_TIME_STEP)
#endif

#ifndef rtmSetTPtr
#define rtmSetTPtr(rtm, val)           ((rtm)->Timing.t = (val))
#endif

extern void flyBall_derivatives();

#endif

