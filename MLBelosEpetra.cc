
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>
#include <Epetra_Map.h>

// Belos provides Krylov solvers
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosMueLuAdapter.hpp>
#include <BelosXpetraAdapterOperator.hpp>

#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>

#include <MueLu.hpp>
#include <MueLu_EpetraOperator.hpp>
#include <MueLu_CreateEpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>

using namespace std;
using namespace Teuchos;


int main(int argc, char *argv[]) {
  // Define default types
  typedef double                                      scalar_type;
  typedef int                                         local_ordinal_type;
  typedef int                                         global_ordinal_type;
  typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;

  // Convenient typedef's
  typedef Epetra_Operator     operator_type;
  typedef Epetra_CrsMatrix    crs_matrix_type;
  typedef Epetra_Vector       vector_type;
  typedef Epetra_MultiVector  multivector_type;
  typedef Epetra_Map          driver_map_type;

  typedef Belos::LinearProblem<scalar_type, multivector_type, operator_type>       linear_problem_type;
  typedef Belos::SolverManager<scalar_type, multivector_type, operator_type>       belos_solver_manager_type;
  typedef Belos::PseudoBlockCGSolMgr<scalar_type, multivector_type, operator_type> belos_pseudocg_manager_type;
  typedef Belos::BlockGmresSolMgr<scalar_type, multivector_type, operator_type>    belos_gmres_manager_type;
  typedef Belos::OperatorTraits<scalar_type,multivector_type,operator_type>        OPT;

  typedef scalar_type         Scalar;
  typedef local_ordinal_type  LocalOrdinal;
  typedef global_ordinal_type GlobalOrdinal;
  typedef node_type           Node;

  typedef MueLu::EpetraOperator         muelu_Epetra_operator_type;
  typedef MueLu::Utilities<scalar_type,local_ordinal_type,global_ordinal_type,node_type>         MueLuUtilities;

# include <MueLu_UseShortNames.hpp>
  typedef Galeri::Xpetra::Problem<Map,CrsMatrixWrap,MultiVector> GaleriXpetraProblem;

  using Teuchos::RCP; // reference count pointers
  using Teuchos::rcp; // reference count pointers

  long initsize  = pow(50,3);
  long finalsize = pow(400,3);
  int base = 50;

  std::ofstream fout;
  struct timeval tim;
  fout.open("MLBelosEpetra.out");

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  RCP< const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
  int mypid = comm->getRank();


  for(long i = initsize; i<=finalsize;){
	  for(int j = 0; j <4; j++){

      // Parameters
	  Teuchos::CommandLineProcessor clp(false);

	  global_ordinal_type            nx = i;
	  Galeri::Xpetra::Parameters<GO> matrixParameters(clp, nx); // manage parameters of the test case
	  Xpetra::Parameters             xpetraParameters(clp);     // manage parameters of xpetra
	  scalar_type tol                = 1e-8;
	  global_ordinal_type maxIts     = i/2;
	  //
	  // Construct the problem
	  //

	  global_ordinal_type indexBase = 0;
	  RCP<const Map>    xpetraMap = MapFactory::Build(Xpetra::UseEpetra, matrixParameters.GetNumGlobalElements(), indexBase, comm);
	  RCP<GaleriXpetraProblem> Pr = Galeri::Xpetra::BuildProblem<scalar_type, local_ordinal_type, global_ordinal_type, Map, CrsMatrixWrap, MultiVector>
	  	  	  	  	  	  	  	    (matrixParameters.GetMatrixType(), xpetraMap, matrixParameters.GetParameterList());
	  RCP<Matrix>         xpetraA = Pr->BuildMatrix();
	  RCP<crs_matrix_type>      A = MueLuUtilities::Op2NonConstEpetraCrs(xpetraA);
	  const driver_map_type   map = MueLuUtilities::Map2EpetraMap(*xpetraMap);

	  // Finish up
	  
	  RCP<multivector_type> X = rcp(new multivector_type(map,1));
	  RCP<multivector_type> B = rcp(new multivector_type(map,1));

	  B->Random();
	  OPT::Apply( *A, *X, *B );
	  X->PutScalar( 0.0 );
	  //MLPreconditioner

	  ParameterList MLList;

	  ML_Epetra::SetDefaults("SA",MLList);
	  MLList.set("output", 10);
	  MLList.set("smoother: type","Chebyshev");
	  MLList.set("smoother: pre or post", "both");
	  MLList.set("coarse: type","Amesos-KLU");
	  
	  ML_Epetra::MultiLevelPreconditioner* MLPrec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);
	  RCP<Belos::EpetraPrecOp> prec = rcp(new Belos::EpetraPrecOp(rcp(MLPrec, false)));

	  //Belos_LinearProblem
	  RCP<linear_problem_type> Problem = rcp(new linear_problem_type(A, X, B));
	  Problem->setRightPrec(prec);
	  Problem->setProblem();
	  
	  //Belos_Solver


	  RCP<ParameterList> belosList = rcp(new ParameterList());
	  belosList->set("Maximum Iterations",    maxIts); // Maximum number of iterations allowed
	  belosList->set("Convergence Tolerance", tol);    // Relative convergence tolerance requested
	  belosList->set("Output Style",          Belos::Brief);
	  belosList->set("Implicit Residual Scaling", "None");
	  RCP<belos_solver_manager_type> solver;
	  solver = rcp(new belos_gmres_manager_type(Problem, belosList));


      gettimeofday (&tim , NULL) ;
	  double t1 = tim.tv_sec+(tim.tv_usec/1e+6) ;

	  solver->solve();

	  gettimeofday (&tim , NULL) ;
	  double t2 = tim.tv_sec+(tim.tv_usec/1e+6) ;

	  int numIterations = solver->getNumIters();


	  Teuchos::Array<typename Teuchos::ScalarTraits<scalar_type>::magnitudeType> normVec(1);
	  multivector_type Ax(B->Map(),1);
	  multivector_type residual(B->Map(),1);
	  A->Apply(*X, residual);
	  residual.Update(1.0, *B, -1.0);
	  residual.Norm2(normVec.getRawPtr());

	  if (mypid == 0) {
		std::cout << "number of iterations = " << numIterations << std::endl;
		std::cout << "||Residual|| = " << normVec[0] << std::endl;
		fout << t2-t1 <<","<< numIterations << ","<< normVec[0] << std::endl;
	  }

	  delete MLPrec;
	}
    base += 50;
	i = pow(base,3);
  }

  #ifdef HAVE_MPI 
  MPI_Finalize() ;
  #endif
  return 0;
}
