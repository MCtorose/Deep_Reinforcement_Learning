#include <stdio.h>
#include "flyBall.h"

static flyBall flyBall_Obj;
void rt_OneStep(void);
void rt_OneStep(void)
{
  static boolean_T OverrunFlag{ false };

  if (OverrunFlag) {
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
  while ((rtmGetErrorStatus(flyBall_Obj.getRTM()) == (nullptr)) &&
         !rtmGetStopRequested(flyBall_Obj.getRTM())) {
    rt_OneStep();
  }

  flyBall_Obj.terminate();
  return 0;
}
