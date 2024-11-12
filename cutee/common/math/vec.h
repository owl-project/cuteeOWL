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

#include "../cutee-common.h"
#include "constants.h"
#include <iostream>
#include <math.h>
#include <algorithm>

namespace cutee {
  namespace common {

    template<typename T> struct long_type_of { typedef T type; };
    template<> struct long_type_of<int32_t>  { typedef int64_t  type; };
    template<> struct long_type_of<uint32_t> { typedef uint64_t type; };
  
    template<typename T, int N>
    struct CUTEE_INTERFACE vec_t { T t[N]; };


    template<typename ScalarTypeA, typename ScalarTypeB> struct BinaryOpResultType;

    // Binary Result type: scalar type with itself always returns same type
    template<typename ScalarType>
    struct BinaryOpResultType<ScalarType,ScalarType> { typedef ScalarType type; };

    template<> struct BinaryOpResultType<int,float> { typedef float type; };
    template<> struct BinaryOpResultType<float,int> { typedef float type; };
    template<> struct BinaryOpResultType<unsigned int,float> { typedef float type; };
    template<> struct BinaryOpResultType<float,unsigned int> { typedef float type; };

    template<> struct BinaryOpResultType<int,double> { typedef double type; };
    template<> struct BinaryOpResultType<double,int> { typedef double type; };
    template<> struct BinaryOpResultType<unsigned int,double> { typedef double type; };
    template<> struct BinaryOpResultType<double,unsigned int> { typedef double type; };
  
    // ------------------------------------------------------------------
    // vec1 - not really a vector, but makes a scalar look like a
    // vector, so we can use it in, say, box1f
    // ------------------------------------------------------------------
    template<typename T>
    struct CUTEE_INTERFACE vec_t<T,1> {
      enum { dims = 1 };
      typedef T scalar_t;
    
      inline vec_t() {}
      inline vec_t(const T &v) : v(v) {}

      /*! assignment operator */
      inline vec_t<T,1> &operator=(const vec_t<T,1> &other) {
        this->v = other.v;
        return *this;
      }
    
      /*! construct 2-vector from 2-vector of another type */
      template<typename OT>
        inline explicit vec_t(const vec_t<OT,1> &o) : v(o.v) {}
    
      inline T &operator[](size_t dim) {
        assert(dim == 0);
        return x;
        // return (&x)[dim];
      }
      inline const T &operator[](size_t dim) const
      {
        assert(dim == 0);
        return x;
        // return (&x)[dim];
      }

      union {
        T v;
        T x; //!< just to allow all vec types to use x,y,z,w,...
      };
    };
 
    // ------------------------------------------------------------------
    // vec2
    // ------------------------------------------------------------------
    template<typename T>
    struct CUTEE_INTERFACE vec_t<T,2> {
      enum { dims = 2 };
      typedef T scalar_t;
    
      inline vec_t() {}
      inline vec_t(const T &t) : x(t), y(t) {}
      inline vec_t(const T &x, const T &y) : x(x), y(y) {}

      /*! assignment operator */
      inline vec_t<T,2> &operator=(const vec_t<T,2> &other) {
        this->x = other.x;
        this->y = other.y;
        return *this;
      }
    
      /*! construct 2-vector from 2-vector of another type */
      template<typename OT>
        inline explicit vec_t(const vec_t<OT,2> &o) : x((T)o.x), y((T)o.y) {}
    
      inline T &operator[](size_t dim) { return (&x)[dim]; }
      inline const T &operator[](size_t dim) const { return (&x)[dim]; }
    
      union {
        struct { T x, y; };
        struct { T s, t; };
        struct { T u, v; };
      };
    };

    // ------------------------------------------------------------------
    // vec3
    // ------------------------------------------------------------------
    template<typename T>
    struct CUTEE_INTERFACE vec_t<T,3> {
      enum { dims = 3 };
      typedef T scalar_t;
    
      inline //
        vec_t(const vec_t &) = default;
      inline vec_t() {}
      inline vec_t(const T &t) : x(t), y(t), z(t) {}
      inline vec_t(const T &_x, const T &_y, const T &_z) : x(_x), y(_y), z(_z) {}
      inline explicit vec_t(const vec_t<T,4> &v);
      /*! construct 3-vector from 3-vector of another type */
      template<typename OT>
        inline explicit vec_t(const vec_t<OT,3> &o) : x((T)o.x), y((T)o.y), z((T)o.z) {}

      /*! swizzle ... */
      inline vec_t<T,3> yzx() const { return vec_t<T,3>(y,z,x); }
    
      /*! assignment operator */
      inline vec_t<T,3> &operator=(const vec_t<T,3> &other) {
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;
        return *this;
      }
    
      inline T &operator[](size_t dim) { return (&x)[dim]; }
      inline const T &operator[](size_t dim) const { return (&x)[dim]; }

      template<typename OT, typename Lambda>
        static inline vec_t<T,3> make_from(const vec_t<OT,3> &v, const Lambda &lambda)
      { return vec_t<T,3>(lambda(v.x),lambda(v.y),lambda(v.z)); }
    
      union {
        struct { T x, y, z; };
        struct { T r, s, t; };
        struct { T u, v, w; };
      };
    };

    // ------------------------------------------------------------------
    // vec3a
    // ------------------------------------------------------------------
    template<typename T>
    struct CUTEE_INTERFACE CUTEE_ALIGN(16) vec3a_t : public vec_t<T,3> {
      inline vec3a_t() {}
      inline vec3a_t(const T &t) : vec_t<T,3>(t) {}
      inline vec3a_t(const T &x, const T &y, const T &z) : vec_t<T,3>(x,y,z) {}

      template<typename OT>
        inline vec3a_t(const vec_t<OT,3> &v) : vec_t<T,3>(v.x,v.y,v.z) {}
    
      T a;
      // add one elemnet for 'forced' alignment
    };
  
    // ------------------------------------------------------------------
    // vec4
    // ------------------------------------------------------------------
    template<typename T>
    struct CUTEE_INTERFACE vec_t<T,4> {
      enum { dims = 4 };
      typedef T scalar_t;
    
      inline vec_t() {}

      inline vec_t(const T &t)
        : x(t), y(t), z(t), w(t)
      {}
      inline vec_t(const vec_t<T,3> &xyz, const T &_w)
        : x(xyz.x), y(xyz.y), z(xyz.z), w(_w)
      {}
      inline vec_t(const T &_x, const T &_y, const T &_z, const T &_w)
        : x(_x), y(_y), z(_z), w(_w)
      {}
    
      /*! construct 3-vector from 3-vector of another type */
      template<typename OT>
        inline explicit vec_t(const vec_t<OT,4> &o)
        : x((T)o.x), y((T)o.y), z((T)o.z), w((T)o.w)
        {}
      inline vec_t(const vec_t<T,4> &o) : x(o.x), y(o.y), z(o.z), w(o.w) {}

      /*! assignment operator */
      inline vec_t<T,4> &operator=(const vec_t<T,4> &other) {
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;
        this->w = other.w;
        return *this;
      }
    
      inline T &operator[](size_t dim) { return (&x)[dim]; }
      inline const T &operator[](size_t dim) const { return (&x)[dim]; }

      template<typename OT, typename Lambda>
        static inline vec_t<T,4> make_from(const vec_t<OT,4> &v,
                                                    const Lambda &lambda)
      { return vec_t<T,4>(lambda(v.x),lambda(v.y),lambda(v.z),lambda(v.w)); }
    
      T x, y, z, w;
    };

    template<typename T>
    inline vec_t<T,3>::vec_t(const vec_t<T,4> &v)
      : x(v.x), y(v.y), z(v.z)
    {}

    // =======================================================
    // default functions
    // =======================================================

    template<typename T>
    inline typename long_type_of<T>::type area(const vec_t<T,2> &v)
    { return (typename long_type_of<T>::type)(v.x)*(typename long_type_of<T>::type)(v.y); }

  
    template<typename T>
    inline typename long_type_of<T>::type volume(const vec_t<T,3> &v)
    { return
        (typename long_type_of<T>::type)(v.x)*
        (typename long_type_of<T>::type)(v.y)*
        (typename long_type_of<T>::type)(v.z);
    }

    template<typename T>
    inline typename long_type_of<T>::type volume(const vec_t<T,4> &v)
    { return
        (typename long_type_of<T>::type)(v.x)*
        (typename long_type_of<T>::type)(v.y)*
        (typename long_type_of<T>::type)(v.z)*
        (typename long_type_of<T>::type)(v.w);
    }

    template<typename T>
    inline typename long_type_of<T>::type area(const vec_t<T,3> &v)
    { return
        T(2)*((typename long_type_of<T>::type)(v.x)*v.y+
              (typename long_type_of<T>::type)(v.y)*v.z+
              (typename long_type_of<T>::type)(v.z)*v.x);
    }



    /*! vector cross product */
    template<typename T>
    inline vec_t<T,3> cross(const vec_t<T,3> &a, const vec_t<T,3> &b)
    {
      return vec_t<T,3>(a.y*b.z-b.y*a.z,
                        a.z*b.x-b.z*a.x,
                        a.x*b.y-b.x*a.y);
    }

    /*! vector cross product */
    template<typename T>
    inline T dot(const vec_t<T,2> &a, const vec_t<T,2> &b)
    {
      return a.x*b.x + a.y*b.y;
    }

    /*! vector cross product */
    template<typename T>
    inline T dot(const vec_t<T,3> &a, const vec_t<T,3> &b)
    {
      return a.x*b.x + a.y*b.y + a.z*b.z;
    }
    
    /*! vector cross product */
    template<typename T>
    inline vec_t<T,3> normalize(const vec_t<T,3> &v)
    {
      return v * cutee::common::polymorphic::rsqrt(dot(v,v));
    }

    /*! vector cross product */
    template<typename T>
    inline T length(const vec_t<T,3> &v)
    {
      return cutee::common::polymorphic::sqrt(dot(v,v));
    }

    template<typename T>
    inline std::ostream &operator<<(std::ostream &o, const vec_t<T,1> &v)
    {
      o << "(" << v.x << ")";
      return o;
    }
  
    template<typename T>
    inline std::ostream &operator<<(std::ostream &o, const vec_t<T,2> &v)
    {
      o << "(" << v.x << "," << v.y << ")";
      return o;
    }
  
    template<typename T>
    inline std::ostream &operator<<(std::ostream &o, const vec_t<T,3> &v)
    {
      o << "(" << v.x << "," << v.y << "," << v.z << ")";
      return o;
    }

    template<typename T>
    inline std::ostream &operator<<(std::ostream &o, const vec_t<T,4> &v)
    {
      o << "(" << v.x << "," << v.y << "," << v.z <<  "," << v.w << ")";
      return o;
    }

    // =======================================================
    // default instantiations
    // =======================================================
  
#define _define_vec_types(T,t)                  \
    using vec2##t = vec_t<T,2>;                 \
    using vec3##t = vec_t<T,3>;                 \
    using vec4##t = vec_t<T,4>;                 \
    using vec3##t##a = vec3a_t<T>;              \
  
    _define_vec_types(bool ,b);
    _define_vec_types(int8_t ,c);
    _define_vec_types(int16_t ,s);
    _define_vec_types(int32_t ,i);
    _define_vec_types(int64_t ,l);
    _define_vec_types(uint8_t ,uc);
    _define_vec_types(uint16_t,us);
    _define_vec_types(uint32_t,ui);
    _define_vec_types(uint64_t,ul);
    _define_vec_types(float,f);
    _define_vec_types(double,d);
  
#undef _define_vec_types

    inline vec_t<bool,3> ge(const vec3f &a, const vec3f &b)
    { return { a.x >= b.x, a.y >= b.y, a.z >= b.z }; }
    
    inline vec_t<bool,3> lt(const vec3f &a, const vec3f &b)
    { return { a.x < b.x, a.y < b.y, a.z < b.z }; }

    inline bool any(vec_t<bool,3> v)
    { return v.x | v.y | v.z; }
    



    // ------------------------------------------------------------------
    // ==
    // ------------------------------------------------------------------

    template<typename T>
    inline bool operator==(const vec_t<T,2> &a, const vec_t<T,2> &b)
    { return a.x==b.x && a.y==b.y; }

    template<typename T>
    inline bool operator==(const vec_t<T,3> &a, const vec_t<T,3> &b)
    { return a.x==b.x && a.y==b.y && a.z==b.z; }

    template<typename T>
    inline bool operator==(const vec_t<T,4> &a, const vec_t<T,4> &b)
    { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
  
    // ------------------------------------------------------------------
    // !=
    // ------------------------------------------------------------------
  
    template<typename T, int N>
    inline bool operator!=(const vec_t<T,N> &a, const vec_t<T,N> &b)
    { return !(a==b); }


    // ------------------------------------------------------------------
    // comparison operators returning result _vector_
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // not (!)
    // ------------------------------------------------------------------

    template<typename T>
    inline auto nt(const vec_t<T,2> &a)
      -> vec_t<decltype(!a.x),2>
    { return { !a.x, !a.y }; }

    template<typename T>
    inline auto nt(const vec_t<T,3> &a)
      -> vec_t<decltype(!a.x),3>
    { return { !a.x, !a.y, !a.z }; }

    template<typename T>
    inline auto nt(const vec_t<T,4> &a)
      -> vec_t<decltype(!a.x),4>
    { return { !a.x, !a.y, !a.z, !a.w }; }

    // ------------------------------------------------------------------
    // eq (==)
    // ------------------------------------------------------------------

    template<typename T>
    inline auto eq(const vec_t<T,2> &a, const vec_t<T,2> &b)
      -> vec_t<decltype(a.x==b.x),2>
    { return { a.x==b.x, a.y==b.y }; }

    template<typename T>
    inline auto eq(const vec_t<T,3> &a, const vec_t<T,3> &b)
      -> vec_t<decltype(a.x==b.x),3>
    { return { a.x==b.x, a.y==b.y, a.z==b.z }; }

    template<typename T>
    inline auto eq(const vec_t<T,4> &a, const vec_t<T,4> &b)
      -> vec_t<decltype(a.x==b.x),4>
    { return { a.x==b.x, a.y==b.y, a.z==b.z, a.w==b.w }; }

    // ------------------------------------------------------------------
    // neq (!=)
    // ------------------------------------------------------------------

    template<typename T, int N>
    inline auto neq(const vec_t<T,N> &a, const vec_t<T,N> &b)
      -> decltype(nt(eq(a,b)))
    { return nt(eq(a,b)); }

   
    
    // ------------------------------------------------------------------
    // reduce
    // ------------------------------------------------------------------

    template<typename T, int N>
    inline bool any(const vec_t<T,N> &a)
    { for (int i=0;i<N;++i) if (a[i]) return true; return false; }

    template<typename T, int N>
    inline bool all(const vec_t<T,N> &a)
    { for (int i=0;i<N;++i) if (!a[i]) return false; return true; }

    // template<typename T>
    // inline bool any(const vec_t<T,3> &a)
    // { return a[i] | b[i] | c[i]; }

    // ------------------------------------------------------------------
    // select
    // ------------------------------------------------------------------

    template<typename T>
    inline vec_t<T,2> select(const vec_t<bool,2> &mask,
                                      const vec_t<T,2> &a,
                                      const vec_t<T,2> &b)
    { return { mask.x?a.x:b.x, mask.y?a.y:b.y }; }

    template<typename T>
    inline vec_t<T,3> select(const vec_t<bool,3> &mask,
                                      const vec_t<T,3> &a,
                                      const vec_t<T,3> &b)
    { return { mask.x?a.x:b.x, mask.y?a.y:b.y, mask.z?a.z:b.z }; }

    template<typename T>
    inline vec_t<T,4> select(const vec_t<bool,4> &mask,
                                      const vec_t<T,4> &a,
                                      const vec_t<T,4> &b)
    { return { mask.x?a.x:b.x, mask.y?a.y:b.y, mask.z?a.z:b.z }; }

    template<typename T, int N>
    inline vec_t<T,N> select(const vec_t<bool,N> &mask,
                                      const vec_t<T,N> &a,
                                      const vec_t<T,N> &b)
    {
      vec_t<T,N> res;
      for (int i=0; i<N; ++i)
        res[i] = mask[i]?a[i]:b[i];
      return res;
    }
  


    // =======================================================
    // vector specializations of those scalar functors
    // =======================================================

    // =======================================================
    // vector specializations of those scalar functors
    // =======================================================

    template<typename T, int N> inline
    bool any_less_than(const vec_t<T,N> &a, const vec_t<T,N> &b);

    template<typename T, int N> inline
    bool all_less_than(const vec_t<T,N> &a, const vec_t<T,N> &b);

    template<typename T, int N> inline
    bool any_greater_or_equal(const vec_t<T,N> &a, const vec_t<T,N> &b);

    template<typename T, int N> inline
    bool any_less_than(const vec_t<T,N> &a, const vec_t<T,N> &b);

    // ----------- specialization for 2 -----------
    template<typename T> inline
    bool any_less_than(const vec_t<T,2> &a, const vec_t<T,2> &b)
    { return (a.x < b.x) | (a.y < b.y); }
  
    template<typename T> inline
    bool all_less_than(const vec_t<T,2> &a, const vec_t<T,2> &b)
    { return (a.x < b.x) & (a.y < b.y); }
  
    template<typename T> inline
    bool any_greater_than(const vec_t<T,2> &a, const vec_t<T,2> &b)
    { return (a.x > b.x) | (a.y > b.y); }

    template<typename T> inline
    bool any_greater_or_equal(const vec_t<T,2> &a, const vec_t<T,2> &b)
    { return (a.x >= b.x) | (a.y >= b.y); }

    // ----------- specialization for 3 -----------
    template<typename T> inline
    bool any_less_than(const vec_t<T,3> &a, const vec_t<T,3> &b)
    { return (a.x < b.x) | (a.y < b.y) | (a.z < b.z); }
  
    template<typename T> inline
    bool all_less_than(const vec_t<T,3> &a, const vec_t<T,3> &b)
    { return (a.x < b.x) & (a.y < b.y) & (a.z < b.z); }
  
    template<typename T> inline
    bool any_greater_than(const vec_t<T,3> &a, const vec_t<T,3> &b)
    { return (a.x > b.x) | (a.y > b.y) | (a.z > b.z); }

    template<typename T> inline
    bool any_greater_or_equal(const vec_t<T,3> &a, const vec_t<T,3> &b)
    { return (a.x >= b.x) | (a.y >= b.y) | (a.z >= b.z); }

    // ----------- specialization for 4 -----------
    template<typename T> inline
    bool any_less_than(const vec_t<T,4> &a, const vec_t<T,4> &b)
    { return (a.x < b.x) | (a.y < b.y) | (a.z < b.z) | (a.w < b.w); }
  
    template<typename T> inline
    bool all_less_than(const vec_t<T,4> &a, const vec_t<T,4> &b)
    { return (a.x < b.x) & (a.y < b.y) & (a.z < b.z) & (a.w < b.w); }
  
    template<typename T> inline
    bool any_greater_than(const vec_t<T,4> &a, const vec_t<T,4> &b)
    { return (a.x > b.x) | (a.y > b.y) | (a.z > b.z) | (a.w > b.w); }

    template<typename T> inline
    bool any_greater_or_equal(const vec_t<T,4> &a, const vec_t<T,4> &b)
    { return (a.x >= b.x) | (a.y >= b.y) | (a.z >= b.z) | (a.w >= b.w); }

    // -------------------------------------------------------
    // unary functors
    // -------------------------------------------------------

    template<typename T>
    inline T clamp(const T &val, const T &lo, const T &hi)
    { return min(hi,max(lo,val)); }
  
    template<typename T>
    inline T clamp(const T &val, const T &hi)
    { return clamp(val,(T)0,hi); }
  
#define _define_float_functor(func)                                     \
    template<typename T> inline vec_t<T,2> func(const vec_t<T,2> &v) \
    { return vec_t<T,2>(cutee::common::func(v.x),cutee::common::func(v.y)); } \
                                                                        \
    template<typename T> inline vec_t<T,3> func(const vec_t<T,3> &v) \
    { return vec_t<T,3>(cutee::common::func(v.x),cutee::common::func(v.y),cutee::common::func(v.z)); } \
                                                                        \
    template<typename T> inline vec_t<T,4> func(const vec_t<T,4> &v) \
    { return vec_t<T,4>(cutee::common::func(v.x),cutee::common::func(v.y),cutee::common::func(v.z),cutee::common::func(v.w)); } \

    _define_float_functor(rcp)
    _define_float_functor(sin)
    _define_float_functor(cos)
    _define_float_functor(abs)
    _define_float_functor(saturate)
  
#undef _define_float_functor
  
    // -------------------------------------------------------
    // binary functors
    // -------------------------------------------------------

#define _define_binary_functor(fct)                                     \
    template<typename T>                                                \
    inline vec_t<T,1> fct(const vec_t<T,1> &a, const vec_t<T,1> &b) \
    {                                                                   \
      return vec_t<T,1>(fct(a.x,b.x));                                  \
    }                                                                   \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,2> fct(const vec_t<T,2> &a, const vec_t<T,2> &b) \
    {                                                                   \
      return vec_t<T,2>(fct(a.x,b.x),                                   \
                        fct(a.y,b.y));                                  \
    }                                                                   \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,3> fct(const vec_t<T,3> &a, const vec_t<T,3> &b) \
    {                                                                   \
      return vec_t<T,3>(fct(a.x,b.x),                                   \
                        fct(a.y,b.y),                                   \
                        fct(a.z,b.z));                                  \
    }                                                                   \
                                                                        \
    template<typename T1, typename T2>                                  \
    inline vec_t<typename BinaryOpResultType<T1,T2>::type,3>   \
    fct(const vec_t<T1,3> &a, const vec_t<T2,3> &b)                     \
    {                                                                   \
      return vec_t<typename BinaryOpResultType<T1,T2>::type,3>          \
        (fct(a.x,b.x),                                                  \
         fct(a.y,b.y),                                                  \
         fct(a.z,b.z));                                                 \
    }                                                                   \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,4> fct(const vec_t<T,4> &a, const vec_t<T,4> &b) \
    {                                                                   \
      return vec_t<T,4>(fct(a.x,b.x),                                   \
                        fct(a.y,b.y),                                   \
                        fct(a.z,b.z),                                   \
                        fct(a.w,b.w));                                  \
    }                                                                   \

  
    _define_binary_functor(divRoundUp)
    _define_binary_functor(min)
    _define_binary_functor(max)
#undef _define_binary_functor




  

    // -------------------------------------------------------
    // binary operators
    // -------------------------------------------------------
#define _define_operator(op)                                            \
    /* vec op vec */                                                    \
    template<typename T>                                                \
    inline vec_t<T,2> operator op(const vec_t<T,2> &a,         \
                                           const vec_t<T,2> &b)         \
    { return vec_t<T,2>(a.x op b.x, a.y op b.y); }                      \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,3> operator op(const vec_t<T,3> &a,         \
                                           const vec_t<T,3> &b)         \
    { return vec_t<T,3>(a.x op b.x, a.y op b.y, a.z op b.z); }          \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,4> operator op(const vec_t<T,4> &a,         \
                                           const vec_t<T,4> &b)         \
    { return vec_t<T,4>(a.x op b.x,a.y op b.y,a.z op b.z,a.w op b.w); } \
                                                                        \
    /* vec op scalar */                                                 \
    template<typename T>                                                \
    inline vec_t<T,2> operator op(const vec_t<T,2> &a,         \
                                           const T &b)                  \
    { return vec_t<T,2>(a.x op b, a.y op b); }                          \
                                                                        \
    template<typename T1, typename T2>                                  \
    inline vec_t<typename BinaryOpResultType<T1,T2>::type,3>   \
    operator op(const vec_t<T1,3> &a, const T2 &b)                      \
    { return vec_t<typename BinaryOpResultType<T1,T2>::type,3>          \
        (a.x op b, a.y op b, a.z op b);                                 \
    }                                                                   \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,4> operator op(const vec_t<T,4> &a,         \
                                           const T &b)                  \
    { return vec_t<T,4>(a.x op b, a.y op b, a.z op b, a.w op b); }      \
                                                                        \
    /* scalar op vec */                                                 \
    template<typename T>                                                \
    inline vec_t<T,2> operator op(const T &a,                  \
                                           const vec_t<T,2> &b)         \
    { return vec_t<T,2>(a op b.x, a op b.y); }                          \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,3> operator op(const T &a,                  \
                                           const vec_t<T,3> &b)         \
    { return vec_t<T,3>(a op b.x, a op b.y, a op b.z); }                \
                                                                        \
    template<typename T>                                                \
    inline vec_t<T,4> operator op(const T &a,                  \
                                           const vec_t<T,4> &b)         \
    { return vec_t<T,4>(a op b.x, a op b.y, a op b.z, a op b.w); }      \
                                                                        \
                                                                        \
    
    _define_operator(*);
    _define_operator(/);
    _define_operator(%);
    _define_operator(+);
    _define_operator(-);
  
#undef _define_operator




    // -------------------------------------------------------
    // unary operators
    // -------------------------------------------------------

    template<typename T>
    inline vec_t<T,2> operator-(const vec_t<T,2> &v)
    { return vec_t<T,2>(-v.x, -v.y); }
  
    template<typename T>
    inline vec_t<T,2> operator+(const vec_t<T,2> &v)
    { return vec_t<T,2>(v.x, v.y); }

    template<typename T>
    inline vec_t<T,3> operator-(const vec_t<T,3> &v)
    { return vec_t<T,3>(-v.x, -v.y, -v.z); }
  
    template<typename T>
    inline vec_t<T,3> operator+(const vec_t<T,3> &v)
    { return vec_t<T,3>(v.x, v.y, v.z); }



    // -------------------------------------------------------
    // binary op-assign operators
    // -------------------------------------------------------
#define  _define_op_assign_operator(operator_op,op)                     \
    /* vec op vec */                                                    \
    template<typename T, typename OT>                                   \
    inline vec_t<T,2> &operator_op(vec_t<T,2> &a,              \
                                            const vec_t<OT,2> &b)       \
    {                                                                   \
      a.x op (T)b.x;                                                    \
      a.y op (T)b.y;                                                    \
      return a;                                                         \
    }                                                                   \
                                                                        \
    template<typename T, typename OT>                                   \
    inline vec_t<T,3> &operator_op(vec_t<T,3> &a,              \
                                            const vec_t<OT,3> &b)       \
    {                                                                   \
      a.x op (T)b.x;                                                    \
      a.y op (T)b.y;                                                    \
      a.z op (T)b.z;                                                    \
      return a;                                                         \
    }                                                                   \
                                                                        \
    template<typename T, typename OT>                                   \
    inline vec_t<T,4> &operator_op(vec_t<T,4> &a,              \
                                            const vec_t<OT,4> &b)       \
    {                                                                   \
      a.x op (T)b.x;                                                    \
      a.y op (T)b.y;                                                    \
      a.z op (T)b.z;                                                    \
      a.w op (T)b.w;                                                    \
      return a;                                                         \
    }                                                                   \
                                                                        \
    /* vec op scalar */                                                 \
    template<typename T, typename OT>                                   \
    inline vec_t<T,2> &operator_op(vec_t<T,2> &a,              \
                                            const OT &b)                \
    { a.x op (T)b; a.y op (T)b; return a; }                             \
                                                                        \
    template<typename T, typename OT>                                   \
    inline vec_t<T,3> &operator_op(vec_t<T,3> &a,              \
                                            const OT &b)                \
    { a.x op (T)b; a.y op (T)b; a.z op (T)b; return a; }                \
                                                                        \
    template<typename T, typename OT>                                   \
    inline vec_t<T,4> &operator_op(vec_t<T,4> &a,              \
                                            const OT &b)                \
    { a.x op (T)b; a.y op (T)b; a.z op (T)b; a.w op (T)b; return a; }   \
    
    _define_op_assign_operator(operator*=,*=);
    _define_op_assign_operator(operator/=,/=);
    _define_op_assign_operator(operator+=,+=);
    _define_op_assign_operator(operator-=,-=);
  
#undef _define_op_assign_operator


    template<typename T>
    T reduce_min(const vec_t<T,1> &v) { return v.x; }
    template<typename T>
    T reduce_min(const vec_t<T,2> &v) { return min(v.x,v.y); }
    template<typename T>
    T reduce_min(const vec_t<T,3> &v) { return min(min(v.x,v.y),v.z); }
    template<typename T>
    T reduce_min(const vec_t<T,4> &v) { return min(min(v.x,v.y),min(v.z,v.w)); }
    template<typename T>
    T reduce_max(const vec_t<T,2> &v) { return max(v.x,v.y); }
    template<typename T>
    T reduce_max(const vec_t<T,3> &v) { return max(max(v.x,v.y),v.z); }
    template<typename T>
    T reduce_max(const vec_t<T,4> &v) { return max(max(v.x,v.y),max(v.z,v.w)); }


    template<typename T, int N>
    vec_t<T,3> madd(const vec_t<T,N> &a, const vec_t<T,N> &b, const vec_t<T,N> &c)
    {
      return a*b + c;
    }


    template<typename T, int N>
    int arg_max(const vec_t<T,N> &v)
    {
      int biggestDim = 0;
      for (int i=1;i<N;i++)
        if ((v[i]) > (v[biggestDim])) biggestDim = i;
      return biggestDim;
    }

    template<typename T, int N>
    int arg_min(const vec_t<T,N> &v)
    {
      int biggestDim = 0;
      for (int i=1;i<N;i++)
        if ((v[i]) < (v[biggestDim])) biggestDim = i;
      return biggestDim;
    }
  

    // less, for std::set, std::map, etc
    template<typename T>
    bool operator<(const vec_t<T,2> &a, const vec_t<T,2> &b)
    {
      if (a.x < b.x) return true;
      if (a.x == b.x && a.y < b.y) return true;
      return false;
    }
    template<typename T>
    bool operator<(const vec_t<T,3> &a, const vec_t<T,3> &b)
    {
      if (a.x < b.x) return true;
      if (a.x == b.x && a.y < b.y) return true;
      if (a.x == b.x && a.y == b.y && a.z < b.z) return true;
      return false;
    }
    template<typename T>
    bool operator<(const vec_t<T,4> &a, const vec_t<T,4> &b)
    {
      if (a.x < b.x) return true;
      if (a.x == b.x && a.y < b.y) return true;
      if (a.x == b.x && a.y == b.y && a.z < b.z) return true;
      if (a.x == b.x && a.y == b.y && a.z == b.z && a.w < b.w) return true;
      return false;
    }

    /*! helper function that creates a semi-random color from an ID */
    inline vec3f randomColor(int i)
    {
      int r = unsigned(i)*13*17 + 0x234235;
      int g = unsigned(i)*7*3*5 + 0x773477;
      int b = unsigned(i)*11*19 + 0x223766;
      return vec3f((r&255)/255.f,
                   (g&255)/255.f,
                   (b&255)/255.f);
    }

    /*! helper function that creates a semi-random color from an ID */
    inline vec3f randomColor(size_t idx)
    {
      unsigned int r = (unsigned int)(idx*13*17 + 0x234235);
      unsigned int g = (unsigned int)(idx*7*3*5 + 0x773477);
      unsigned int b = (unsigned int)(idx*11*19 + 0x223766);
      return vec3f((r&255)/255.f,
                   (g&255)/255.f,
                   (b&255)/255.f);
    }

    /*! helper function that creates a semi-random color from an ID */
    template<typename T>
    inline vec3f randomColor(const T *ptr)
    {
      return randomColor((size_t)ptr);
    }

    inline float sqrt(const float v) { return sqrtf(v); }
    inline vec2f sqrt(const vec2f v) { return vec2f(sqrtf(v.x),sqrtf(v.y)); }
    inline vec3f sqrt(const vec3f v) { return vec3f(sqrtf(v.x),sqrtf(v.y),sqrtf(v.z)); }
    inline vec4f sqrt(const vec4f v) { return vec4f(sqrtf(v.x),sqrtf(v.y),sqrtf(v.z),sqrtf(v.w)); }

    
  } // ::cutee::common
} // ::cutee


