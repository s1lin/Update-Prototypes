
#include "AztecOO_config.h"
// Epetra provides distributed sparse linear algebra
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>
#include <Epetra_Map.h>

#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>

#include "AztecOO.h"

#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>

using namespace std;
using namespace Teuchos;


int main(int argc, char *argv[]) {

  typedef double                                      scalar_type;
  typedef int                                         local_ordinal_type;
  typedef int                                         global_ordinal_type;
  typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;

  typedef Epetra_Operator     operator_type;
  typedef Epetra_CrsMatrix    crs_matrix_type;
  typedef Epetra_Vector       vector_type;
  typedef Epetra_MultiVector  multivector_type;
  typedef Epetra_Map          driver_map_type;

  long initsize  = pow(50,3);
  long finalsize = pow(400,3);
  int base = 50;

  std::ofstream fout;
  struct timeval tim;
  fout.open("MLAztecOOEpetra.out");

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  RCP< const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
  int mypid = comm->getRank();

  cout << comm <<endl;

  for(long i = initsize; i<=finalsize;){
	for(int j = 0; j <4; j++){

      // Parameters

	  global_ordinal_type nx = i;
	  Galeri::Xpetra::Parameters<GO> matrixParameters(clp, nx); // manage parameters of the test case
	  Xpetra::Parameters             xpetraParameters(clp);     // manage parameters of xpetra

	  global_ordinal_type maxIts            = 1000;
	  scalar_type tol                       = 1e-8;
	  std::string solverOptionsFile         = "amg.xml";
	  std::string krylovSolverType          = "gmres";
	  //
	  // Construct the problem
	  //

	  global_ordinal_type indexBase = 0;
	  RCP<const Map>    xpetraMap = MapFactory::Build(Xpetra::UseEpetra, matrixParameters.GetNumGlobalElements(), indexBase, comm);
	  RCP<GaleriXpetraProblem> Pr = Galeri::Xpetra::BuildProblem<scalar_type, local_ordinal_type, global_ordinal_type, Map, CrsMatrixWrap, MultiVector>
	  	  	  	  	  	  	  	    (matrixParameters.GetMatrixType(), xpetraMap, matrixParameters.GetParameterList());
	  RCP<Matrix>         xpetraA = Pr->BuildMatrix();
	  RCP<crs_matrix_type>      A = MueLuUtilities::Op2NonConstEpetraCrs(xpetraA);
	  const driver_map_type map = MueLuUtilities::Map2EpetraMap(*xpetraMap);
	  // Finish up

	  Epetra_Vector x(map);
	  Epetra_Vector b(map);
	  b.Random(); // Fill b with random values

	  Epetra_LinearProblem problem(&A.get(), &x, &b);
	  AztecOO solver(problem);

	  solver.SetAztecOption(AZ_precond, AZ_Jacobi);


	  ParameterList MLList;

	  ML_Epetra::SetDefaults("SA",MLList);
	  MLList.set("output", 10);
	  MLList.set("max levels",5);
	  MLList.set("increasing or decreasing","increasing");
	  MLList.set("aggregation: type", "Uncoupled");
	  MLList.set("smoother: type","symmetric Gauss-Seidel");
	  MLList.set("smoother: pre or post", "both");
	  MLList.set("coarse: type","Amesos-KLU");

	  ML_Epetra::MultiLevelPreconditioner* MLPrec =
		new ML_Epetra::MultiLevelPreconditioner(A, MLList);


	  solver.SetPrecOperator(MLPrec);
	  solver.SetAztecOption(AZ_solver, AZ_gmres);
	  solver.SetAztecOption(AZ_output, 32);

      gettimeofday (&tim , NULL) ;
	  double t1 = tim.tv_sec+(tim.tv_usec/1e+6) ;

	  solver.Iterate(i/2, 1.0E-8);

	  gettimeofday (&tim , NULL) ;
	  double t2 = tim.tv_sec+(tim.tv_usec/1e+6) ;

	  delete MLPrec;


	  cout << "Solver performed " << solver.NumIters() << " iterations." << endl
		   << "Norm of true residual = " << solver.TrueResidual() << endl;
      fout << t2-t1 <<","<< solver.NumIters() << ","<< solver.TrueResidual()  << std::endl;
	}
    base += 50;
	i = pow(base,3);
  }
  #ifdef HAVE_MPI
   MPI_Finalize() ;
  #endif
  return 0;
}
