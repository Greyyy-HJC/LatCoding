#include "lib.h"


EXPORT(Gauge_fix,{
    
    PyObject* _metadata;
    PyObject* _args;
    PyObject* _maxiter;
    PyObject* _prec;
    PyObject* _fourier = Py_False; // add by Jinchen
    PyObject* _orthog = PyLong_FromLong(-1); // default: -1 (Landau)
    PyObject* _ret;

    // if (!PyArg_ParseTuple(args, "OOO", &_args, &_maxiter, &_prec)) {
    //   std::cout << "Error reading arguments" << std::endl;
    //   return NULL;
    // }
    if (!PyArg_ParseTuple(args, "OOO|OO", &_args, &_maxiter, &_prec, &_fourier, &_orthog)) { // add by Jinchen
      std::cout << "Error reading arguments" << std::endl;
      return NULL;
    }

    auto grid = get_pointer<GridCartesian>(_args,"U_grid");

    // Do I need this?: FFT theFFT(grid);

    // Lattice< iLorentzColourMatrix<vComplexD> > U(grid);
    LatticeGaugeFieldD U(grid);
    for (int mu=0;mu<Nd;mu++) {
        auto l = get_pointer<cgpt_Lattice_base>(_args,"U",mu);
        auto& Umu = compatible<iColourMatrix<vComplexD>>(l)->l;
        PokeIndex<LorentzIndex>(U,Umu,mu);
    }

    // Now do gauge fixing

    Real alpha=0.1;
    int maxiter;
    Real prec;
    cgpt_convert(_prec, prec);
    cgpt_convert(_maxiter, maxiter);
    LatticeColourMatrixD xform1(grid);

    // Fourier acceleration, add by Jinchen
    bool use_fourier = PyObject_IsTrue(_fourier);
    int orthog_dir;
    cgpt_convert(_orthog, orthog_dir);

    // FourierAcceleratedGaugeFixer<PeriodicGimplR>::SteepestDescentGaugeFix(U,xform1,alpha,maxiter,prec,prec,false,3);
    FourierAcceleratedGaugeFixer<PeriodicGimplR>::SteepestDescentGaugeFix(
      U, xform1, alpha, maxiter, prec, prec, use_fourier, orthog_dir
    ); // add by Jinchen

    // Transfrom back to stuff that gpt can deal with
    std::vector< cgpt_Lattice_base* > U_prime(4);
    for (int mu=0;mu<4;mu++) {
      auto lat = new cgpt_Lattice< iColourMatrix< vComplexD > >(grid);
      lat->l = PeekIndex<LorentzIndex>(U,mu);
      U_prime[mu] = lat;
    }
    auto trafo = new cgpt_Lattice< iColourMatrix< vComplexD> >(grid);
    trafo->l = xform1;

    // return
    vComplexD vScalar = 0; // TODO: grid->to_decl()
    // return Py_BuildValue("([(l,[i,i,i,i],s,s,[N,N,N,N])],N)", grid, grid->_gdimensions[0], grid->_gdimensions[1], 
    //    grid->_gdimensions[2], grid->_gdimensions[3], get_prec(vScalar).c_str(), "full", U_prime[0]->to_decl(), 
    //    U_prime[1]->to_decl(), U_prime[2]->to_decl(), U_prime[3]->to_decl(), _metadata);
        return Py_BuildValue("([(l,[i,i,i,i],s,s,[N,N,N,N])],N)", grid, grid->_gdimensions[0], grid->_gdimensions[1], 
       grid->_gdimensions[2], grid->_gdimensions[3], get_prec(vScalar).c_str(), "full", U_prime[0]->to_decl(), 
       U_prime[1]->to_decl(), U_prime[2]->to_decl(), U_prime[3]->to_decl(), trafo->to_decl());
});
