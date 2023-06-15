//cdesensamblador.cpp
#include "cdesensamblador.h"

void CDesensamblador::FetchOperandos(CEntero16b &dir,int md1,CEntero16b &op1,
                                     int md2,CEntero16b &op2)
{
    CEntero16b registroauxiliar;

    //PRIMER OPERANDO
    if (md1!=0)
    {
        conf->Memoria()->Leer(dir,registroauxiliar);
        if (md1==MD_INMEDIATO || md1==MD_MEMORIA)
        {
            //Leemos la palabra entera
            op1=registroauxiliar;
        }
        else
        {
            //Leemos el byte superior
            if(md1==MD_REGISTRO || md1==MD_INDIRECTO)
            {
                //Solo interesan los 4 bits inferiores del byte superior
                op1=CEntero16b((registroauxiliar.Valor()/256)%16);
            }
            else
            {
                op1=CEntero16b(registroauxiliar.Valor()/256);
            }
        }
        dir++;
    }
    else
    {
        //No hay primer operando
        op1=CEntero16b(0);
    }

    //SEGUNDO OPERANDO
    if (md2!=0)
    {
        if (md1==MD_INMEDIATO || md1==MD_MEMORIA ||
            md2==MD_INMEDIATO || md2==MD_MEMORIA)
            {
            //Hay algun operando que necesita una palabra completa
            //Tenemos que leer otra palabra
            conf->Memoria()->Leer(dir,registroauxiliar);
            dir++;
        }
        //Tenemos el operando en el registro auxiliar
        //Recuperamos el byte inferior
        if (md2==MD_INMEDIATO || md2==MD_MEMORIA)
        {
            op2=registroauxiliar;
        }
        else
        {
            if (md2==MD_REGISTRO || md2==MD_INDIRECTO)
            {
                op2=CEntero16b(registroauxiliar.Valor()%16);
            }
            else
            {
                op2=CEntero16b(registroauxiliar.Valor()%256);
            }
        }
    }
    else
    {
        //No hay segundo operando
        op2=CEntero16b(0);
    }
}
//---------------------------------------------------------------------------
CDesensamblador::CDesensamblador(CConfiguracion *configuracion)
{
    conf=configuracion;
    return;
}
//---------------------------------------------------------------------------
void CDesensamblador::Desensamblar(CEntero16b &dirinicio,CCadena &instruccion)
{
    int codop;
    int mdir1;
    int mdir2;
    CEntero16b op1;
    CEntero16b op2;

    CEntero16b registroauxiliar;
    CCadena nombreregistro;

    CEntero16b direccionsiguiente;

    //Leemos la posicion de memoria actual
    conf->Memoria()->Leer(dirinicio,registroauxiliar);
    dirinicio++;
    direccionsiguiente=dirinicio;

    //Recuperamos el codigo de operacion y los modos de direccionamiento
    codop=registroauxiliar.Valor() / 64; //10 primeros bits
    mdir1=(registroauxiliar.Valor() / 8) % 8; //bits 11-13
    mdir2=(registroauxiliar.Valor() % 8); //bits 14-16

    FetchOperandos(dirinicio,mdir1,op1,mdir2,op2);

    if(ComprobarInstruccion(codop,mdir1,op1.Valor(),mdir2,op2.Valor())!=0)
    {
        //Si la instruccion no esta bien formada
        //la mostramos como No Implementada
        codop=-1;
        mdir1=0;
        mdir2=0;
        dirinicio=direccionsiguiente;
    }

    switch(codop){
        case NOP :    instruccion=CCadena((char *) "NOP ");
                      break;
        case HALT :   instruccion=CCadena((char *) "HALT ");
                      break;
        case MOVE :   instruccion=CCadena((char *) "MOVE ");
                      break;
        case PUSH :   instruccion=CCadena((char *) "PUSH ");
                      break;
        case POP :    instruccion=CCadena((char *) "POP ");
                      break;
        case ADD :    instruccion=CCadena((char *) "ADD ");
                      break;
        case SUB :    instruccion=CCadena((char *) "SUB ");
                      break;
        case MUL :    instruccion=CCadena((char *) "MUL ");
                      break;
        case DIV :    instruccion=CCadena((char *) "DIV ");
                      break;
        case MOD :    instruccion=CCadena((char *) "MOD ");
                      break;
        case INC :    instruccion=CCadena((char *) "INC ");
                      break;
        case DEC :    instruccion=CCadena((char *) "DEC ");
                      break;
        case NEG :    instruccion=CCadena((char *) "NEG ");
                      break;
        case CMP :    instruccion=CCadena((char *) "CMP ");
                      break;
        case AND :    instruccion=CCadena((char *) "AND ");
                      break;
        case OR :     instruccion=CCadena((char *) "OR ");
                      break;
        case XOR :    instruccion=CCadena((char *) "XOR ");
                      break;
        case NOT :    instruccion=CCadena((char *) "NOT ");
                      break;
        case BR :     instruccion=CCadena((char *) "BR ");
                      break;
        case BZ :     instruccion=CCadena((char *) "BZ ");
                      break;
        case BNZ :    instruccion=CCadena((char *) "BNZ ");
                      break;
        case BP :     instruccion=CCadena((char *) "BP ");
                      break;
        case BN :     instruccion=CCadena((char *) "BN ");
                      break;
        case BV :     instruccion=CCadena((char *) "BV ");
                      break;
        case BNV :    instruccion=CCadena((char *) "BNV ");
                      break;
        case BC :     instruccion=CCadena((char *) "BC ");
                      break;
        case BNC :    instruccion=CCadena((char *) "BNC ");
                      break;
        case BE :     instruccion=CCadena((char *) "BE ");
                      break;
        case BO :     instruccion=CCadena((char *) "BO ");
                      break;
        case CALL :   instruccion=CCadena((char *) "CALL ");
                      break;
        case RET :    instruccion=CCadena((char *) "RET ");
                      break;
        case INCHAR : instruccion=CCadena((char *) "INCHAR ");
                      break;
        case ININT :  instruccion=CCadena((char *) "ININT ");
                      break;
        case INSTR :  instruccion=CCadena((char *) "INSTR ");
                      break;
        case WRCHAR : instruccion=CCadena((char *) "WRCHAR ");
                      break;
        case WRINT :  instruccion=CCadena((char *) "WRINT ");
                      break;
        case WRSTR :  instruccion=CCadena((char *) "WRSTR ");
                      break;
        default :     instruccion=CCadena((char *) S_CADENA_001); //*NO IMPLEMENTADA* 
                      return;
    }

    //Primer operando
    switch(mdir1)
    {
        case MD_INMEDIATO :

            instruccion=instruccion+"#";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op1.Decimal();
                           break;
                case 16 : instruccion=instruccion+op1.Hexadecimal();
                          break;
            }
            break;

        case MD_REGISTRO :

            instruccion=instruccion+".";
            conf->BancoRegistros()->LeerNombreRegistro(op1.Valor(),
                                                       nombreregistro);
            instruccion=instruccion+nombreregistro;
            break;

        case MD_MEMORIA :

            instruccion=instruccion+"/";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op1.DecimalSinSigno();
                          break;
                case 16 : instruccion=instruccion+op1.Hexadecimal();
                          break;
            }
            break;

        case MD_INDIRECTO :

            instruccion=instruccion+"[.";
            conf->BancoRegistros()->LeerNombreRegistro(op1.Valor(),
                                                       nombreregistro);
            instruccion=instruccion+nombreregistro;
            instruccion=instruccion+"]";
            break;

        case MD_RELATIVO_IX :

            instruccion=instruccion+"#";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op1.DecimalCorto();
                          break;
                case 16 : instruccion=instruccion+op1.HexadecimalCorto();
                          break;
            }
            instruccion=instruccion+"[.IX]";
            break;

        case MD_RELATIVO_IY :

            instruccion=instruccion+"#";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op1.DecimalCorto();
                          break;
                case 16 : instruccion=instruccion+op1.HexadecimalCorto();
                          break;
            }
            instruccion=instruccion+"[.IY]";
            break;

        case MD_RELATIVO_PC :

            instruccion=instruccion+"$";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op1.DecimalCorto();
                          break;
                case 16 : instruccion=instruccion+op1.HexadecimalCorto();
                          break;
            }
            break;
    }
    if(mdir2!=MD_NO_OPERANDO)
    {
        instruccion=instruccion+",";
    }
    //Segundo operando
    switch(mdir2)
    {
        case MD_INMEDIATO :

             instruccion=instruccion+"#";
             switch(conf->BaseNumerica())
             {
                 case 10 : instruccion=instruccion+op2.Decimal();
                           break;
                 case 16 : instruccion=instruccion+op2.Hexadecimal();
                           break;
             }
             break;

        case MD_REGISTRO :

            instruccion=instruccion+".";
            conf->BancoRegistros()->LeerNombreRegistro(op2.Valor(),
                                                       nombreregistro);
            instruccion=instruccion+nombreregistro;
            break;

        case MD_MEMORIA :

            instruccion=instruccion+"/";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op2.DecimalSinSigno();
                          break;
                case 16 : instruccion=instruccion+op2.Hexadecimal();
                          break;
            }
            break;

        case MD_INDIRECTO :

            instruccion=instruccion+"[.";
            conf->BancoRegistros()->LeerNombreRegistro(op2.Valor(),
                                                       nombreregistro);
            instruccion=instruccion+nombreregistro;
            instruccion=instruccion+"]";
            break;

        case MD_RELATIVO_IX :

            instruccion=instruccion+"#";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op2.DecimalCorto();
                          break;
                case 16 : instruccion=instruccion+op2.HexadecimalCorto();
                          break;
            }
            instruccion=instruccion+"[.IX]";
            break;

        case MD_RELATIVO_IY :

            instruccion=instruccion+"#";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op2.DecimalCorto();
                          break;
                case 16 : instruccion=instruccion+op2.HexadecimalCorto();
                          break;
            }
            instruccion=instruccion+"[.IY]";
            break;

        case MD_RELATIVO_PC :

            instruccion=instruccion+"$";
            switch(conf->BaseNumerica())
            {
                case 10 : instruccion=instruccion+op2.DecimalCorto();
                          break;
                case 16 : instruccion=instruccion+op2.HexadecimalCorto();
                          break;
            }
            break;
    }
}
