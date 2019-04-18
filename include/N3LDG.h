#ifndef N3LDG_ALL
#define N3LDG_ALL

#include "profiler.h"
#include "Graph.h"
#include "Node.h"
#include "Alphabet.h"
#include "NRMat.h"
#include "MyLib.h"
#include "Metric.h"
#include "AtomicOP.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "BucketOP.h"
#include "LookupTable.h"
#include "Param.h"
#include "SparseParam.h"
#include "ModelUpdate.h"
#include "CheckGrad.h"
#include "Pooling.h"
#include "Concat.h"
#include "Windowlized.h"
#include "UniOP.h"
#include "LSTM1.h"
#include "SoftMaxLoss.h"
#include "AttentionHelp.h"
#include "Attention.h"
#include "SparseOP.h"
#include "MaxProbabilityLoss.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif

#endif
