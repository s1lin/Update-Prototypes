
#include <iostream>

// Epetra provides distributed sparse linear algebra
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

// Galeri
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>

// MueLu main header: include most common header files in one line
#include <MueLu.hpp>
#include <MueLu_EpetraOperator.hpp>
#include <MueLu_CreateEpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

using Teuchos::RCP;
using Teuchos::rcp;
using namespace std;

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

  typedef MueLu::EpetraOperator         muelu_Epetra_operator_type;
  typedef MueLu::Utilities<scalar_type,local_ordinal_type,global_ordinal_type,node_type>         MueLuUtilities;

  typedef Belos::LinearProblem<scalar_type, multivector_type, operator_type>       linear_problem_type;
  typedef Belos::SolverManager<scalar_type, multivector_type, operator_type>       belos_solver_manager_type;
  typedef Belos::PseudoBlockCGSolMgr<scalar_type, multivector_type, operator_type> belos_pseudocg_manager_type;
  typedef Belos::BlockGmresSolMgr<scalar_type, multivector_type, operator_type>    belos_gmres_manager_type;
  typedef Belos::OperatorTraits<scalar_type,multivector_type,operator_type>        OPT;

  typedef Teuchos::ScalarTraits<scalar_type>   ST;


  //MueLu_UseShortNames.hpp wants these typedefs.
  typedef scalar_type         Scalar;
  typedef local_ordinal_type  LocalOrdinal;
  typedef global_ordinal_type GlobalOrdinal;
  typedef node_type           Node;
  //typedef Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> Map;
# include <MueLu_UseShortNames.hpp>


  typedef Galeri::Xpetra::Problem<Map,CrsMatrixWrap,MultiVector> GaleriXpetraProblem;

  using Teuchos::RCP; // reference count pointers
  using Teuchos::rcp; // reference count pointers

  //
  // MPI initialization using Teuchos
  //

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  RCP< const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
  int mypid = comm->getRank();

  Teuchos::CommandLineProcessor clp(false);

  // Parameters

  global_ordinal_type nx = 50;
  Galeri::Xpetra::Parameters<GO> matrixParameters(clp, nx); // manage parameters of the test case
  Xpetra::Parameters             xpetraParameters(clp);     // manage parameters of xpetra

  global_ordinal_type maxIts            = 1000;
  scalar_type tol                       = 1e-8;
  std::string solverOptionsFile         = "amg.xml";
  std::string krylovSolverType          = "gmres";

  clp.setOption("xmlFile",    &solverOptionsFile, "XML file containing MueLu solver parameters");
  clp.setOption("maxits",     &maxIts,            "maximum number of Krylov iterations");
  clp.setOption("tol",        &tol,               "tolerance for Krylov solver");
  clp.setOption("krylovType", &krylovSolverType,  "cg or gmres solver");

  switch (clp.parse(argc, argv)) {
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS;
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:          break;
  }

  if (xpetraParameters.GetLib() == Xpetra::UseEpetra) {
    throw std::invalid_argument("This example only supports Epetra.");
  }

  ParameterList mueluParams;
  Teuchos::updateParametersFromXmlFile(solverOptionsFile, Teuchos::inoutArg(mueluParams));

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

  A->OptimizeStorage();


  //
  // Set up linear problem Ax = b and associate preconditioner with it.
  //
  RCP<multivector_type> X = rcp(new multivector_type(map,1));
  RCP<multivector_type> B = rcp(new multivector_type(map,1));


  X->PutScalar((scalar_type) 0.0);
  B->Random();

  //
  // Construct a multigrid preconditioner
  //
  // Multigrid Hierarchy
  A->Apply(*X,*B);
  RCP<muelu_Epetra_operator_type> M = MueLu::CreateEpetraPreconditioner(A, mueluParams);

  RCP<linear_problem_type> Problem = rcp(new linear_problem_type(A, X, B));
  Problem->setRightPrec(M);
  Problem->setProblem();

  //
  // Set up Krylov solver and iterate.
  //
  RCP<ParameterList> belosList = rcp(new ParameterList());
  belosList->set("Maximum Iterations",    maxIts); // Maximum number of iterations allowed
  belosList->set("Convergence Tolerance", tol);    // Relative convergence tolerance requested
  belosList->set("Verbosity",             Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
  belosList->set("Output Frequency",      1);
  belosList->set("Output Style",          Belos::Brief);
  belosList->set("Implicit Residual Scaling", "None");
  RCP<belos_solver_manager_type> solver;
  if (krylovSolverType == "cg")
    solver = rcp(new belos_pseudocg_manager_type(Problem, belosList));
  else if (krylovSolverType == "gmres")
    solver = rcp(new belos_gmres_manager_type(Problem, belosList));
  else
    throw std::invalid_argument("bad Krylov solver type");


  solver->solve();
  int numIterations = solver->getNumIters();
    std::cout<<std::endl<<"here\n";

  Teuchos::Array<typename Teuchos::ScalarTraits<scalar_type>::magnitudeType> normVec(1);
  multivector_type Ax(B->Map(),1);
  multivector_type residual(B->Map(),1);
  A->ApplyInverse(*X, residual);
  residual.Update(1.0, *B, -1.0);
  residual.Norm2(normVec.getRawPtr());
  if (mypid == 0) {
    std::cout << "number of iterations = " << numIterations << std::endl;
    std::cout << "||Residual|| = " << normVec[0] << std::endl;
  }

  return EXIT_SUCCESS;
}

