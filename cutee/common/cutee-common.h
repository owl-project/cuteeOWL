// ======================================================================== //
// Copyright 2018-2022 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/* copied from OWL project, and put into new namespace to avoid naming conflicts.*/

#pragma once

#include <QtGlobal>
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
# define WITH_QT5
#else
# define WITH_QT6
#endif

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <math.h> // using cmath causes issues under Windows

#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#ifdef __GNUC__
#include <execinfo.h>
#include <sys/time.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#if !defined(WIN32)
#include <signal.h>
#endif

#if defined(_MSC_VER)
#  define CUTEE_DLL_EXPORT __declspec(dllexport)
#  define CUTEE_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define CUTEE_DLL_EXPORT __attribute__((visibility("default")))
#  define CUTEE_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define CUTEE_DLL_EXPORT
#  define CUTEE_DLL_IMPORT
#endif

// #if 1
# define CUTEE_INTERFACE /* nothing - currently not building any special 'owl.dll' */
// #else
// //#if defined(CUTEE_DLL_INTERFACE)
// #  ifdef owl_EXPORTS
// #    define CUTEE_INTERFACE CUTEE_DLL_EXPORT
// #  else
// #    define CUTEE_INTERFACE CUTEE_DLL_IMPORT
// #  endif
// //#else
// //#  define CUTEE_INTERFACE /*static lib*/
// //#endif
// #endif

//#ifdef __WIN32__
//#define  __PRETTY_FUNCTION__ __FUNCTION__ 
//#endif
#if defined(_MSC_VER)
//&& !defined(__PRETTY_FUNCTION__)
#  define __PRETTY_FUNCTION__ __FUNCTION__
#endif


#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif

#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

// namespace detail {
// inline static std::string backtrace()
// {
// #ifdef __GNUC__
//     static const int max_frames = 16;

//     void* buffer[max_frames] = { 0 };
//     int cnt = ::backtrace(buffer,max_frames);

//     char** symbols = backtrace_symbols(buffer,cnt);

//     if (symbols) {
//       std::stringstream str;
//       for (int n = 1; n < cnt; ++n) // skip the 1st entry (address of this function)
//       {
//         str << symbols[n] << '\n';
//       }
//       free(symbols);
//       return str.str();
//     }
//     return "";
// #else
//     return "not implemented yet";
// #endif
// }

// inline void cuteeRaise_impl(std::string str)
// {
//   fprintf(stderr,"%s\n",str.c_str());
// #ifdef WIN32
//   if (IsDebuggerPresent())
//     DebugBreak();
//   else
//     throw std::runtime_error(str);
// #else
// #ifndef NDEBUG
//   std::string bt = ::detail::backtrace();
//   fprintf(stderr,"%s\n",bt.c_str());
// #endif
//   raise(SIGINT);
// #endif
// }
// }

// #define CUTEE_RAISE(MSG) ::detail::cuteeRaise_impl(MSG);


#define CUTEE_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not implemented")

// #ifdef WIN32
// # define CUTEE_TERMINAL_RED ""
// # define CUTEE_TERMINAL_GREEN ""
// # define CUTEE_TERMINAL_LIGHT_GREEN ""
// # define CUTEE_TERMINAL_YELLOW ""
// # define CUTEE_TERMINAL_BLUE ""
// # define CUTEE_TERMINAL_LIGHT_BLUE ""
// # define CUTEE_TERMINAL_RESET ""
// # define CUTEE_TERMINAL_DEFAULT CUTEE_TERMINAL_RESET
// # define CUTEE_TERMINAL_BOLD ""

// # define CUTEE_TERMINAL_MAGENTA ""
// # define CUTEE_TERMINAL_LIGHT_MAGENTA ""
// # define CUTEE_TERMINAL_CYAN ""
// # define CUTEE_TERMINAL_LIGHT_RED ""
// #else
// # define CUTEE_TERMINAL_RED "\033[0;31m"
// # define CUTEE_TERMINAL_GREEN "\033[0;32m"
// # define CUTEE_TERMINAL_LIGHT_GREEN "\033[1;32m"
// # define CUTEE_TERMINAL_YELLOW "\033[1;33m"
// # define CUTEE_TERMINAL_BLUE "\033[0;34m"
// # define CUTEE_TERMINAL_LIGHT_BLUE "\033[1;34m"
// # define CUTEE_TERMINAL_RESET "\033[0m"
// # define CUTEE_TERMINAL_DEFAULT CUTEE_TERMINAL_RESET
// # define CUTEE_TERMINAL_BOLD "\033[1;1m"

// # define CUTEE_TERMINAL_MAGENTA "\e[35m"
// # define CUTEE_TERMINAL_LIGHT_MAGENTA "\e[95m"
// # define CUTEE_TERMINAL_CYAN "\e[36m"
// # define CUTEE_TERMINAL_LIGHT_RED "\033[1;31m"
// #endif

#ifdef _MSC_VER
# define CUTEE_ALIGN(alignment) __declspec(align(alignment)) 
#else
# define CUTEE_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif



namespace cutee {
  namespace common {

    using std::min;
    using std::max;
    using std::abs;
    inline float saturate(const float &f) { return min(1.f,max(0.f,f)); }

    // inline float abs(float f)      { return fabsf(f); }
    // inline double abs(double f)    { return fabs(f); }
    inline float rcp(float f)      { return 1.f/f; }
    inline double rcp(double d)    { return 1./d; }
  
    inline int32_t divRoundUp(int32_t a, int32_t b) { return (a+b-1)/b; }
    inline uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a+b-1)/b; }
    inline int64_t divRoundUp(int64_t a, int64_t b) { return (a+b-1)/b; }
    inline uint64_t divRoundUp(uint64_t a, uint64_t b) { return (a+b-1)/b; }
  
    using ::sin; // this is the double version
    using ::cos; // this is the double version

    /*! namespace that offers polymorphic overloads of functions like
        sqrt, rsqrt, sin, cos, etc (that vary based on float vs
        double), and that is NOT in a default namespace where ti
        would/could clash with cuda or system-defines of the same name
        - TODO: make sure that cos, sin, abs, etc are also properly
        handled here. */
    namespace polymorphic {
      inline float sqrt(const float f)     { return ::sqrtf(f); }
      inline double sqrt(const double d)   { return ::sqrt(d); }
      
      inline float rsqrt(const float f)    { return 1.f/cutee::common::polymorphic::sqrt(f); }
      inline double rsqrt(const double d)  { return 1./cutee::common::polymorphic::sqrt(d); }
    }
    

#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif
  
    /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
    inline std::string prettyDouble(const double val) {
      const double absVal = abs(val);
      char result[1000];

      if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
      else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
      else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
      else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
      else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
      else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
      else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
      else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
      else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
      else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
      else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
      else osp_snprintf(result,1000,"%f",(float)val);

      return result;
    }
  

    /*! return a nicely formatted number as in "3.4M" instead of
        "3400000", etc, using mulitples of thousands (K), millions
        (M), etc. Ie, the value 64000 would be returned as 64K, and
        65536 would be 65.5K */
    inline std::string prettyNumber(const size_t s)
    {
      char buf[1000];
      if (s >= (1000LL*1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1000.f*1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1000.f*1000.f));
      } else if (s >= (1000LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1000.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    /*! return a nicely formatted number as in "3.4M" instead of
        "3400000", etc, using mulitples of 1024 as in kilobytes,
        etc. Ie, the value 65534 would be 64K, 64000 would be 63.8K */
    inline std::string prettyBytes(const size_t s)
    {
      char buf[1000];
      if (s >= (1024LL*1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
      } else if (s >= (1024LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }
  
    inline double getCurrentTime()
    {
#ifdef _WIN32
      SYSTEMTIME tp; GetSystemTime(&tp);
      /*
         Please note: we are not handling the "leap year" issue.
     */
      size_t numSecsSince2020
          = tp.wSecond
          + (60ull) * tp.wMinute
          + (60ull * 60ull) * tp.wHour
          + (60ull * 60ul * 24ull) * tp.wDay
          + (60ull * 60ul * 24ull * 365ull) * (tp.wYear - 2020);
      return double(numSecsSince2020 + tp.wMilliseconds * 1e-3);
#else
      struct timeval tp; gettimeofday(&tp,nullptr);
      return double(tp.tv_sec) + double(tp.tv_usec)/1E6;
#endif
    }

    inline bool hasSuffix(const std::string &s, const std::string &suffix)
    {
      return s.substr(s.size()-suffix.size()) == suffix;
    }
    
  } // ::cutee::common
} // ::cutee
