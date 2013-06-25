/*
 * vm.h
 *
 *  Created on: 12.06.2013
 *      Author: mrupp
 */

#ifndef VM_H_
#define VM_H_

#include "common/log.h"
#include <vector>
#include "parser_node.h"
#include "common/assert.h"
#include "parser.hpp"

namespace ug{

class VMAdd
{
	std::vector<char> vmBuf;
	std::string m_name;
	std::vector<double> variables;
	size_t m_nrOut, m_nrIn;

	enum VMInstruction
	{
		PUSH_CONSTANT=0,
		PUSH_VAR,
		JMP_IF_FALSE,
		JMP,
		OP_UNARY,
		OP_BINARY,
		ASSIGN,
		OP_RETURN,
		OP_CALL
	};



	void serialize(VMInstruction inst)
	{
		int i=inst;
		serialize(i);
	}


	void deserialize(size_t &p, VMInstruction &instr)
	{
		int i = *((int*)&vmBuf[p]);
		instr = (VMInstruction) i;
		p += sizeof(int);
	}

	void serialize(char c)
	{
		vmBuf.push_back(c);
	}

	void serialize(int d)
	{
		char *tmp = (char*)&d;
		for(size_t i=0; i<sizeof(d); i++)
			serialize(tmp[i]);
	}

	void deserialize(size_t &p, int &i)
	{
		i = *((int*)&vmBuf[p]);
		p += sizeof(int);
	}

	void serialize(double d)
	{
		char *tmp = (char*)&d;
		for(size_t i=0; i<sizeof(d); i++)
			serialize(tmp[i]);
	}

	void deserialize(size_t &p, double &d)
	{
		d = *((double*)&vmBuf[p]);
		p += sizeof(double);
	}


	template<typename T>
	void print_unary(const char *desc, size_t &i)
	{
		T t;
		deserialize(i, t);
		UG_LOG(desc << " " << t << "\n");
	}

	void print_op(size_t &i)
	{
		int op;
		deserialize(i, op);
		switch(op)
		{
		case MATH_COS: 		UG_LOG("cos\n"); break;
		case MATH_SIN: 		UG_LOG("sin\n"); break;
		case MATH_EXP: 		UG_LOG("exp\n"); break;
		case MATH_ABS: 		UG_LOG("abs\n"); break;
		case MATH_LOG: 		UG_LOG("log\n"); break;
		case MATH_LOG10: 	UG_LOG("log10\n"); break;
		case MATH_SQRT: 		UG_LOG("sqrt\n"); break;
		case MATH_FLOOR: 		UG_LOG("floor\n"); break;
		case MATH_CEIL: 		UG_LOG("ceil\n"); break;
		case '+': 		UG_LOG("+\n"); break;
		case '-': 		UG_LOG("-\n"); break;
		case '*': 		UG_LOG("*\n"); break;
		case '/': 		UG_LOG("/\n"); break;
		case '<': 		UG_LOG("<\n"); break;
		case '>': 		UG_LOG(">\n"); break;
		case GE: 		UG_LOG("GE\n"); break;
		case LE: 		UG_LOG("LE\n"); break;
		case NE: 		UG_LOG("NE\n"); break;
		case EQ: 		UG_LOG("EQ\n"); break;
		case AND: 		UG_LOG("AND\n"); break;
		case OR: 		UG_LOG("OR\n"); break;
		case MATH_POW: 		UG_LOG("pow\n"); break;
		case MATH_MIN: 		UG_LOG("min\n"); break;
		case MATH_MAX: 		UG_LOG("max\n"); break;
		}
	}

public:
	VMAdd()
	{
			m_name = "unknown";
	}
	void set_name(std::string name)
	{
		m_name = name;
	}

	void push(double constant)
	{
//		UG_LOG("POS " << get_pos() << "\n");
//		UG_LOG("PUSH_CONSTANT " << constant << "\n");
		serialize((int)PUSH_CONSTANT);
		serialize(constant);
	}

	void push_var(int i)
	{
//		UG_LOG("PUSH_VAR " << i << "\n");
		serialize((int)PUSH_VAR);
		serialize(i);
//		UG_LOG("POS " << get_pos() << "\n");
	}

	int jmp_if_false()
	{
//		UG_LOG("JMP_IF_FALSE\n");
		return jump(JMP_IF_FALSE);
	}

	int jmp()
	{
//		UG_LOG("JMP\n");
		return jump(JMP);
	}

	int get_pos()
	{
		return vmBuf.size();
	}

	void unary(int oper)
	{
//		UG_LOG("UNARY OP " << oper << "\n");
		serialize((int)OP_UNARY);
		serialize(oper);
//		UG_LOG("POS " << get_pos() << "\n");
	}

	void binary(int oper)
	{
//		UG_LOG("BINARY OP " << oper << "\n");
		serialize((int)OP_BINARY);
		serialize(oper);
//		UG_LOG("POS " << get_pos() << "\n");
	}

	void assign(int v)
	{
//		UG_LOG("ASSIGN " << v << "\n");
		serialize((int)ASSIGN);
		serialize(v);
//		UG_LOG("POS " << get_pos() << "\n");
	}

	std::vector<SmartPtr<VMAdd> > subfunctions;
	void call(SmartPtr<VMAdd> subfunction)
	{
		size_t i;
		for(i=0; i<subfunctions.size(); i++)
			if(subfunctions[i] == subfunction)
				break;
		if(i == subfunctions.size())
			subfunctions.push_back(subfunction);
		serialize((int)OP_CALL);
		serialize((int)i);
	}

	void adjust_jmp_pos(int iPos, int jmpPos)
	{
//		UG_LOG("adjusting jmp pos in " << iPos << " to " << jmpPos << "\n");
		int *p = (int *)&vmBuf[iPos];
		*p = jmpPos;
	}

	void ret()
	{
		serialize((int)OP_RETURN);
	}

	void print_short()
	{
		UG_LOG("function " << m_name << ", " << m_nrIn << " Parameters, " << variables.size() << " variables, " << subfunctions.size() << " subfunctions");
	}

	void print()
	{
		print_short();
		UG_LOG("\n");
		for(size_t i=0; i<vmBuf.size(); )
		{
			UG_ASSERT(i<vmBuf.size(), i);
			UG_LOG(i << "	");
			VMInstruction instr;
			deserialize(i, instr);

			switch(instr)
			{
				case PUSH_CONSTANT:
					print_unary<double>("PUSH_CONSTANT", i);
					break;

				case PUSH_VAR:
					print_unary<int>("PUSH_VAR", i);
					break;

				case JMP_IF_FALSE:
					print_unary<int>("JMP_IF_FALSE", i);
					break;
				case JMP:
					print_unary<int>("JMP", i);
					break;

				case OP_UNARY:
				case OP_BINARY:
					print_op(i);
					break;
				case ASSIGN:
					print_unary<int>("ASSIGN", i);
					break;
				case OP_RETURN:
					UG_LOG("RETURN\n");
					break;

				case OP_CALL:
				{
					int varI;
					deserialize(i, varI);

					UG_ASSERT(varI < subfunctions.size(), i);
					UG_LOG("CALL to subfunction " << varI << ": ");
					subfunctions[varI]->print_short();
					UG_LOG("\n");
					break;
				}

				default:
					UG_LOG(((int)instr) << "?\n");
			}
		}
	}

	int jump(VMInstruction instr)
	{
		serialize(instr);
		int jmpPos = get_pos();
		serialize(jmpPos);
//		UG_LOG("jump pos is " << jmpPos << "\n");
		return jmpPos;
	}



	void set_in_out(size_t nrIn,size_t nrOut)
	{
		m_nrOut = nrOut;
		m_nrIn = nrIn;
	}
	void set_nr_of_variables(size_t nr)
	{
		variables.resize(nr);
	}

	void execute_unary(size_t &i, double &v)
	{
		int op;
//		UG_LOG("unary op " << v << "\n");
		deserialize(i, op);
		switch(op)
		{
			case MATH_COS: v = cos(v);	break;
			case MATH_SIN: v = sin(v);	break;
			case MATH_EXP: v = exp(v);	break;
			case MATH_ABS: v = abs(v); 	break;
			case MATH_LOG: v = log(v); 	break;
			case MATH_LOG10: v = log10(v); break;
			case MATH_SQRT:  v = sqrt(v); 	break;
			case MATH_FLOOR: v = floor(v); break;
			case MATH_CEIL: v = ceil(v); break;
		}
	}

	void execute_binary(size_t &i, double *stack, int SP)
	{
		double &a = stack[SP-2];
		double &b = stack[SP-1];
//		UG_LOG("binary op " << a << " op " << b << "\n");

		int op;
		deserialize(i, op);
		switch(op)
		{
			case '+': 	a = b+a;	break;
			case '-': 	a = b-a;	break;
			case '*': 	a = b*a;	break;
			case '/': 	a = b/a;	break;
			case '<': 	a = (b < a) ? 1.0 : 0.0; break;
			case '>': 	a = (b > a) ? 1.0 : 0.0; break;
			case GE: 	a = (b >= a) ? 1.0 : 0.0; break;
			case LE: 	a = (b <= a) ? 1.0 : 0.0; break;
			case NE: 	a = (b != a) ? 1.0 : 0.0; break;
			case EQ: 	a = (b == a) ? 1.0 : 0.0; break;
			case AND: 	a = (a != 0.0 && b != 0.0) ? 1.0 : 0.0; break;
			case OR: 	a = (a != 0 || b != 0) ? 1.0 : 0.0; break;
			case MATH_POW: 	a = pow(b, a); break;
			case MATH_MIN: 	a = (b < a) ? a : b; break;
			case MATH_MAX: 	a = (b > a) ? a : b; break;
		}

	}

	double call(double *stack, int &SP)
	{
//		for(int j=0; j<variables.size(); j++)
//		{UG_LOG("var[[" << j << "] = " << variables[j] << "\n");}

		double varD;
		int varI;
		size_t i=0;
		VMInstruction instr;
		while(1)
		{
//			UG_LOG("IP =  " << i << ", SP = " << SP);
			deserialize(i, instr);
//			UG_LOG("OP =  " << (int)instr << "\n");
//			for(int j=0; j<SP; j++)
//				{UG_LOG("SP[" << j << "] = " << stack[j] << ", ");}
//			UG_LOG("\n");

			switch(instr)
			{
				case PUSH_CONSTANT:

					deserialize(i, varD);
//					UG_LOG("PUSH CONSTANT " << varD << "\n");
					stack[SP++] = varD;
					break;

				case PUSH_VAR:
					deserialize(i, varI);
					stack[SP++] = variables[varI-1];
//					UG_LOG("PUSH VAR " << varI << "\n");
					break;

				case JMP_IF_FALSE:
					deserialize(i, varI);
//					UG_LOG("JMP IF FALSE " << varI << "\n");
					SP--;
					if(stack[SP] == 0.0)
						i = varI;
					break;
				case JMP:
					deserialize(i, varI);
//					UG_LOG("JMP " << varI << "\n");
					i = varI;
					break;

				case OP_UNARY:
					UG_ASSERT(SP>0, SP);
					execute_unary(i, stack[SP-1]);
					break;

				case OP_BINARY:
					UG_ASSERT(SP>1, SP);
					execute_binary(i, stack, SP);
					SP--;
					break;

				case ASSIGN:
					deserialize(i, varI);
					SP--;
//					UG_LOG("ASSIGN " << varI << " = " << stack[SP] << "\n");
					variables[varI-1] = stack[SP];
					break;

				case OP_RETURN:
//					UG_LOG("RETURN\n");
					UG_ASSERT(SP == (int)m_nrOut, "stack pointer is not nrOut =" << m_nrOut << ", instead " << SP << " ?")
					return stack[0];
					break;

				case OP_CALL:
				{
					deserialize(i, varI);
//					UG_LOG("call IP =  " << i << ", SP = " << SP << "\n");
					SmartPtr<VMAdd> sub = subfunctions[varI];
					sub->call_sub(stack, SP);
//					UG_LOG("call IP =  " << i << ", SP = " << SP << "\n");
					break;
				}
				default:
					UG_ASSERT(0, "IP: " << i << " op " << ((int)instr) << " ?\n");
			}
		}
	}

	double call_sub(double *stack, int &SP)
	{
		for(size_t i=0; i<m_nrIn; i++)
			variables[i] = stack[SP+i-1];
		SP -= m_nrIn;
		return call(stack, SP);
	}

	void operator ()(double *ret, double *in)
	{
		double stack[255];
		int SP=0;

		for(size_t i=0;i<m_nrIn; i++)
			variables[i] = in[i];
		call(stack, SP);
		UG_ASSERT(SP == (int)m_nrOut, SP << " != " << m_nrOut);
		for(size_t i=0; i<m_nrOut; i++)
			ret[i] = stack[i];
	}

	double call()
	{
		double stack[255];
		int SP=0;
		return call(stack, SP);
	}

	double call(double a, double b, double c)
	{
		UG_ASSERT(m_nrOut == 1, m_nrOut);
		UG_ASSERT(m_nrIn == 3, m_nrIn);
		variables[0] = a;
		variables[1] = b;
		variables[2] = c;
		return call();
	}

	double call(double a, double b)
	{
		UG_ASSERT(m_nrOut == 1, m_nrOut);
		UG_ASSERT(m_nrIn == 2, m_nrIn);
		variables[0] = a;
		variables[1] = b;
		return call();
	}

	double call(double a)
	{
		UG_ASSERT(m_nrOut == 1, m_nrOut);
		UG_ASSERT(m_nrIn == 1, m_nrIn);
		variables[0] = a;
		return call();
	}
	size_t num_out()
	{
		return m_nrOut;
	}
	size_t num_in()
	{
		return m_nrOut;
	}
};

}
#endif /* VM_H_ */