/*
 * lua_parser_class_create_c.cpp
 *
 *  Created on: 12.06.2013
 *      Author: mrupp
 */

#include "lua_parser_class.h"
#include "common/assert.h"
#include "common/util/string_util.h"

using namespace std;
namespace ug{

int LUAParserClass::createC(nodeType *p, ostream &out, int indent)
{
	if (!p) return 0;
	switch (p->type)
	{
		case typeCon:
			out << p->con.value;
			break;
		case typeId:
			getVar(p->id.i, out);
			break;
		case typeOpr:
			switch (p->opr.oper)
			{
				case IF:
                {
					out << repeat('\t', indent);			out << "if(";
					createC(p->opr.op[0], out, 0);
					out << ")\n";
					out << repeat('\t', indent); 			out << "{\n";
					createC(p->opr.op[1], out, indent+1);
					out << repeat('\t', indent); 			out << "}\n";

                    nodeType *a = p->opr.op[2];
                    while(a != NULL && a->opr.oper == ELSEIF)
                    {
                        out << repeat('\t', indent); 			out << "else if(";
                        createC(a->opr.op[0], out, 0);
                        out << ")\n";
                        out << repeat('\t', indent); 			out << "{\n";
                        createC(a->opr.op[1], out, indent+1);
                        out << repeat('\t', indent); 			out << "}\n";
                        a = a->opr.op[2];
                    }
                    if(a != NULL)
                    {
                        UG_ASSERT(a->opr.oper == ELSE, a->opr.oper);
                        out << repeat('\t', indent); 			out << "else\n";
                        out << repeat('\t', indent); 			out << "{\n";
                        createC(a->opr.op[0], out, indent+1);
                        out << repeat('\t', indent); 			out << "}\n";
                    }

                    break;
                }

				case '=':
                    UG_ASSERT(is_local(p->opr.op[0]->id.i), "global variable " << id2variable[p->opr.op[0]->id.i] << " is read-only");
					out << repeat('\t', indent);
					out << id2variable[p->opr.op[0]->id.i] << " = ";
					createC(p->opr.op[1], out, 0);
					out << ";\n";
					break;

                case 'C':
                {
                    out << "LUA2C_Subfunction_" << id2variable[p->opr.op[0]->id.i] << "(";
                    nodeType *a = p->opr.op[1];
					while(a->type == typeOpr && a->opr.oper == ',')
					{
						createC(a->opr.op[0], out, indent);
						out << ", ";
						a = a->opr.op[1];
					}
					createC(a, out, indent);
					out << ")\n";
                    break;
                }

				case 'R':
				{
					if(returnType == RT_SUBFUNCTION || returnType == RT_NEUMANN)
                    {
                        nodeType *a = p->opr.op[0];
                        if(a->type == typeOpr && a->opr.oper == ',')
                        {
                            UG_ASSERT(0, "subfunctions may not return more then one value");
                            return false;
                        }
                        out << repeat('\t', indent);
                        out << "return ";
                        createC(a, out, indent);
                        out << ";\n";
                    }
                    else if(returnType == RT_DIFFUSION)
                    {
                        const char *rt[] = {"A11", "A12", "A21", "A22"};
                        createRT(p->opr.op[0], out, rt, 4, indent);

                        out << "\n";
                        out << repeat('\t', indent);
                        out << "goto diffusionReturn;\n";
                    }
                    else if(returnType == RT_VELOCITY)
                    {
                        const char *rt[] = {"vx", "vy"};
                        createRT(p->opr.op[0], out, rt, 2, indent);

                        out << "\n";
                        out << repeat('\t', indent);
                        out << "goto velocityReturn;\n";
                    }
                    else if(returnType == RT_DIRICHLET || returnType == RT_SOURCE)
                    {
                        const char *rt[] = {"f"};
                        createRT(p->opr.op[0], out, rt, 1, indent);

                        out << "\n";
                        out << repeat('\t', indent);
						if(returnType == RT_DIRICHLET)
							out << "goto dirichletReturn;\n";
						else
							out << "goto sourceReturn;\n";
                    }
                    else
                    {
                        nodeType *a = p->opr.op[0];
                        int i=0;
                        while(a->type == typeOpr && a->opr.oper == ',')
                        {
                            out << repeat('\t', indent);
                            out << "LUA2C_ret[" << i++ << "] = ";
                            createC(a->opr.op[0], out, indent);
                            out << ";\n";
                            a = a->opr.op[1];
                        }
                        out << repeat('\t', indent);
                        out << "LUA2C_ret[" << i++ << "] = ";
                        createC(a, out, indent);
                        out << ";\n";
                        out << repeat('\t', indent);
                        out << "return 1;\n";
                    }
                    break;
				}

				case TK_FOR:
					out << repeat('\t', indent);
					out << "for(";
					createC(p->opr.op[0], out, 0);
					out << " = ";
					createC(p->opr.op[1], out, 0);
					out << "; ";
					createC(p->opr.op[0], out, 0);
					out << " <= ";
					createC(p->opr.op[2], out, 0);
					out << "; ";
					createC(p->opr.op[0], out, 0);
					out << " += ";
					createC(p->opr.op[3], out, 0);
					out << ")\n";
					out << repeat('\t', indent);
					out << "{\n";
						createC(p->opr.op[4], out, indent+1);
					out << repeat('\t', indent);
					out << "}\n";
					break;

				case TK_BREAK:
					out << repeat('\t', indent);
					out << "break;\n";
					break;

				case UMINUS:
					out << "-(";
					createC(p->opr.op[0], out, 0);
					out << ")";
					break;

                case MATH_PI:
                    out << " MATH_PI ";
                    break;

				case MATH_COS:
                case MATH_SIN:
                case MATH_EXP:
                case MATH_ABS:
                case MATH_LOG:
                case MATH_LOG10:
                case MATH_SQRT:
                case MATH_FLOOR:
                case MATH_CEIL:
                    switch (p->opr.oper)
                    {
                        case MATH_COS: out << "cos("; break;
                        case MATH_SIN: out << "sin("; break;
                        case MATH_EXP: out << "exp("; break;
                        case MATH_ABS: out << "fabs("; break;
                        case MATH_LOG: out << "log("; break;
                        case MATH_LOG10: out << "log10("; break;
                        case MATH_SQRT: out << "sqrt("; break;
                        case MATH_FLOOR: out << "floor("; break;
                        case MATH_CEIL: out << "ceil("; break;
                    }
					createC(p->opr.op[0], out, 0);
					out << ")";
					break;

                case MATH_POW:
                case MATH_MIN:
                case MATH_MAX:
                    switch (p->opr.oper)
                    {
                        case MATH_POW: out << "pow("; break;
                        case MATH_MIN: out << "min("; break;
                        case MATH_MAX: out << "max("; break;
                    }

					createC(p->opr.op[0], out, 0);
					out << ", ";
                    createC(p->opr.op[1], out, 0);
                    out << ")";
					break;

				case ';':
					createC(p->opr.op[0], out, indent);
					createC(p->opr.op[1], out, indent);
					break;

				default:
					out << "(";
					createC(p->opr.op[0], out, 0);
					out << ")";

					switch (p->opr.oper)
					{
						case '+': out << '+';
							break;
						case '-': out << '-';
							break;
						case '*': out << '*';
							break;
						case '/': out << '/';
							break;
						case '<': out << '<';
							break;
						case '>': out << '>';
							break;
						case GE: out << " >= ";
							break;
						case LE: out << " <= ";
							break;
						case NE: out << " != ";
							break;
						case EQ: out << " == ";
							break;
						case AND: out << " && ";
							break;
						case OR: out << " || ";
							break;
					}
					out << "(";
					createC(p->opr.op[1], out, 0);
					out << ")";
			}
	}
	return 0;
}

int LUAParserClass::createC(ostream &out)
{
     // local functions

    set<string> knownFunctions;
    stringstream declarations;
    stringstream definitions;
    if(add_subfunctions(knownFunctions, declarations, definitions) == false)
    {
        UG_LOG("add_subfunctions failed.\n");
        return false;
    }

    out << "#define MATH_PI 3.1415926535897932384626433832795028841971693\n";
    out << "// inline function declarations\n";
    out << declarations.str() << "\n";

    out << "// inline function definitions\n";
    out << definitions.str() << "\n";

    // the function
	out << "int " << name << "(";
	out << "double *LUA2C_ret, "; //[" << numOut << "], ";
	out << "double *LUA2C_in)\n"; //[" << numIn << "])\n";
	nodeType *a = args;

	out << "{\n";

    int i=0;
	while(a->type == typeOpr)
	{
		out << "\tdouble " << id2variable[a->opr.op[0]->id.i] << " = LUA2C_in[" << i++ << "];\n";
		a = a->opr.op[1];
	}
	out << "\tdouble " << id2variable[a->id.i] << " = LUA2C_in[" << i++ << "];\n";

	//------ local variables --------
	out << "\t// local variables:\n";
	for(map<string, size_t>::iterator it = variables.begin(); it != variables.end(); ++it)
		if(is_local((*it).second))
			out << "\tdouble " << (*it).first << ";\n";


	out << "\t// code:\n";
	for(size_t i=0; i<nodes.size(); i++)
		createC(nodes[i], out, 1);
	out << "}\n";
	return 0;
}


int LUAParserClass::createC_inline(ostream &out)
{
    out << "inline ";
    declare(out);
    out << "\n{\n";

	//------ local variables --------
	out << "\t// local variables:\n";
	for(map<string, size_t>::iterator it = variables.begin(); it != variables.end(); ++it)
		if(is_local((*it).second))
			out << "\tdouble " << (*it).first << ";\n";

	out << "\t// code:\n";
	for(size_t i=0; i<nodes.size(); i++)
		createC(nodes[i], out, 1);
	out << "}\n";
	return true;
}

int LUAParserClass::addfunctionC(string name, set<string> &knownFunctions, stringstream &declarations, stringstream &definitions)
{
    //UG_LOG("adding " << name << "\n");
    if(knownFunctions.find(name) != knownFunctions.end()) return true;
    knownFunctions.insert(name);

    LUAParserClass parser;
    if(parser.parse_luaFunction(name.c_str()) == false)
        return false;

    if(parser.num_out() != 1)
    {
        UG_LOG("ERROR in LUA2C for LUA function " << name << ":  subfunction must have exactly one return value (not " << parser.num_out() << ")\n");
        return false;
    }

    parser.returnType = RT_SUBFUNCTION;

    parser.declare(declarations); declarations << ";\n";

    parser.createC_inline(definitions);

    parser.add_subfunctions(knownFunctions, declarations, definitions);

    return true;
}


int LUAParserClass::declare(ostream &out)
{
    out << "double LUA2C_Subfunction_" << name << "(";
	nodeType *a = args;
	while(a->type == typeOpr)
	{
		out << "double " << id2variable[a->opr.op[0]->id.i] << ", ";
		a = a->opr.op[1];
	}
	out << "double " << id2variable[a->id.i] << ")";
    return true;
}

}
