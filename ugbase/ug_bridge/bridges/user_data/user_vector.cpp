

#include "../../ug_bridge.h"
#include "common/common.h"
#include "lib_discretization/lib_discretization.h"
#include "../../../ug_script/ug_script.h"

#include <sstream>
#include <string>

namespace ug
{
namespace bridge
{

template <int dim>
class ConstUserVector
{
	public:
		ConstUserVector() {set_all_entries(0.0);}

		void set_all_entries(number val) { m_Vector = val;}

		void set_entry(size_t i, number val)
		{
			m_Vector[i] = val;
		}

		void print() const
		{
			UG_LOG("ConstUserVector:" << m_Vector << "\n");
		}

		void operator() (MathVector<dim>& v, const MathVector<dim>& x, number time = 0.0)
		{
			v = m_Vector;
		}

	protected:
		MathVector<dim> m_Vector;
};

template <int dim>
class LuaUserVector
{
	public:
		LuaUserVector()
		{
			m_L = ug::script::GetDefaultLuaState();
		}

		void set_lua_callback(const char* luaCallback)
		{
		//	Todo: Work on function pointer directly
			m_callbackName = luaCallback;
		}

		void operator() (MathVector<dim>& v, const MathVector<dim>& x, number time = 0.0)
		{
			lua_getglobal(m_L, m_callbackName);
			for(size_t i = 0; i < dim; ++i)
				lua_pushnumber(m_L, x[i]);
			lua_pushnumber(m_L, time);

			if(lua_pcall(m_L, dim + 1, dim, 0) != 0)
			{
				UG_LOG("error running diffusion callback " << m_callbackName << ": "
								<< lua_tostring(m_L, -1));
				throw(int(0));
			}

			int counter = -1;
			for(size_t i = 0; i < dim; ++i){
					v[i] = luaL_checknumber(m_L, counter--);
			}

			lua_pop(m_L, dim);
		}

	protected:
		const char* m_callbackName;
		lua_State*	m_L;
};


template <int dim, typename TUserVector>
class UserVectorProvider : public IUserVectorProvider<dim>
{
	public:
	//	Functor Type
		typedef typename IUserVectorProvider<dim>::functor_type functor_type;

	//	return functor
		virtual functor_type get_functor() const {return m_UserVector;}

	//	set user Vector
		void set_functor(const TUserVector& userVector) {m_UserVector = userVector;}

	protected:
		TUserVector	m_UserVector;
};

template <int dim>
void RegisterUserVector(Registry& reg, const char* parentGroup)
{
	const char* grp = parentGroup;

//	Functor
	{
	//	ConstUserVector
		{
			typedef ConstUserVector<dim> T;
			static stringstream ss; ss << "ConstUserVector" << dim << "d";
			reg.add_class_<T>(ss.str().c_str(), grp)
				.add_constructor()
				.add_method("set_all_entries", &T::set_all_entries)
				.add_method("set_entry", &T::set_entry)
				.add_method("print", &T::print);
		}

	//	LuaUserVector
		{
			typedef LuaUserVector<dim> T;
			stringstream ss; ss << "LuaUserVector" << dim << "d";
			reg.add_class_<T>(ss.str().c_str(), grp)
				.add_constructor()
				.add_method("set_lua_callback", &T::set_lua_callback);
		}
	}

//	UserVectorProvider
	{
	//	Base class
		{
			stringstream ss; ss << "IUserVectorProvider" << dim << "d";
			reg.add_class_<IUserVectorProvider<dim> >(ss.str().c_str(), grp);
		}

	//	Const Vector Provider
		{
			typedef UserVectorProvider<dim, ConstUserVector<dim> > T;
			stringstream ss; ss << "ConstUserVectorProvider" << dim << "d";
			reg.add_class_<T, IUserVectorProvider<dim> >(ss.str().c_str(), grp)
				.add_constructor()
				.add_method("set_functor", &T::set_functor);
		}

	//	Lua Vector
		{
			typedef UserVectorProvider<dim, LuaUserVector<dim> > T;
			stringstream ss; ss << "LuaUserVectorProvider" << dim << "d";
			reg.add_class_<T, IUserVectorProvider<dim> >(ss.str().c_str(), grp)
				.add_constructor()
				.add_method("set_functor", &T::set_functor);
		}
	}
}

void RegisterUserVector(Registry& reg, const char* parentGroup)
{
	RegisterUserVector<1>(reg, parentGroup);
	RegisterUserVector<2>(reg, parentGroup);
	RegisterUserVector<3>(reg, parentGroup);
}

} // end namespace
} // end namepace
