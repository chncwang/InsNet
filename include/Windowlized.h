#ifndef WINDOWLIZED
#define WINDOWLIZED

/*
*  Windowlized.h:
*  a contextual builder, concatenate x[-c]...x[0]...x[c] together
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Concat.h"
#include "Graph.h"

class WindowBuilder {
  public:
    int _context;
    int _window;
    int _nSize;
    int _inDim;
    int _outDim;

    vector<ConcatNode> _outputs;
    BucketNode _bucket;


  public:
    WindowBuilder() {
        clear();
    }

    ~WindowBuilder() {
        clear();
    }


    inline void resize(int maxsize) {
        _outputs.resize(maxsize);
    }

    inline void clear() {
        _outputs.clear();
        _context = 0;
        _window = 0;
        _nSize = 0;
        _inDim = 0;
        _outDim = 0;
    }


    inline void init(int inDim, int context) {
        _context = context;
        _window = 2 * _context + 1;
        _inDim = inDim;
        _outDim = _window * _inDim;
        int maxsize = _outputs.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _outputs[idx].init(_outDim, -1); // dropout is not supported here
        }
        _bucket.init(_inDim, -1);
    }



  public:
    inline void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for windowlized operation" << std::endl;
            return;
        }

        _nSize = x.size();

        vector<PNode> in_nodes(_window);
        _bucket.forward(cg, 0);
        for (int idx = 0; idx < _nSize; idx++) {
            int offset = 0;
            in_nodes[offset++] = x[idx];
            for (int j = 1; j <= _context; j++) {
                in_nodes[offset++] = idx - j >= 0 ? x[idx - j] : &_bucket;
                in_nodes[offset++] = idx + j < _nSize ? x[idx + j] : &_bucket;
            }
            _outputs[idx].forward(cg, in_nodes);
        }
    }

};


#endif
