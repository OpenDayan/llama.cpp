#pragma once

// #define ENABLE_PARAMS_CHECK

#if defined(ENABLE_PARAMS_CHECK)
#    define DAYAN_PARAMS_IF(condition) if (condition)
#    define DAYAN_PARAMS_ERROR() dayan_proc_exit(-22)
#else
#    define DAYAN_PARAMS_IF(condition) if constexpr (false)
#    define DAYAN_PARAMS_ERROR() __builtin_unreachable()
#endif
