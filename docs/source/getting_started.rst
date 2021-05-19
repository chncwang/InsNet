N3LDG++ Tutorial
============================

In this topic, we will demonstrate how to train a model using N3LDG++ from scratch, taking the conversation model as an example.

Before getting started, note that N3LDG++ uses procedural style computation APIs. Thus Unlike PyTorch, there is no concept of "module" and parameters are decoupled from models, making it straightforward when reusing the same parameters in different layers or places of the neural network.

Getting Started
------------------

Firstly, we define model parameters as follows:

.. code-block:: c++

    struct ModelParams : public n3ldg_plus::TunableParamCollection
    // To specify parameters that can be optimized, ModelParams needs to be a child class of
    // n3ldg_plus::TunableParamCollection.
    {
        // Assume word embeddings are shared between the encoder and decoder.
        n3ldg_plus::Embedding<Param> embedding;
        n3ldg_plus::TransformerEncoderParams encoder;
        n3ldg_plus::TransformerDecoderParams decoder;

        // Since C++ does not support reflection mechanism natively, we need to specify each
        // parameter, except for those fixed during training.
        std::vector<n3ldg_plus::TunableParam *> tunableComponents() override {
            return {&embedding, &encoder, &decoder};
        }

        void init(const std::vector<std::string> &word_list, int dim, int layer, int head,
                int max_position) {
            n3ldg_plus::Vocab vocab;
            vocab.init(word_list);
            embedding.init(vocab, dim);
            // The default dimension of Relu's intermmediate layers is 4 x dim.
            encoder.init(layer, dim, head, max_position);
            decoder.init(layer, dim, head, max_position);
        }
    };

Next, we declare and initialize the parameters.

.. code-block:: c++

    ModelParams model_params;
    model_params.init(word_list, 512, 3, 8, 512);

Then assuming we choose Adam to optimize parameters, we can create the optimizer as follows:

.. code-block:: c++

    // Create the Adam optimzer with [lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-10, l2_reg=0].
    n3ldg_plus::Optimzer *optimizer = new n3ldg_plus::AdamOptimizer(
            model_params.tunableParams(), 1e-4, 0.9, 0.999, 1e-10, 0);

During training, we can change the learning rate as follows:

.. code-block:: c++

    using n3ldg_plus::dtype;
    if (iteration > 40000) {
        dtype lr = 1e-4 * std::sqrt(40000.0f / iteration);
        optimizer->setLearningRate(lr);
    }

Supposing that we have already loaded and randomly shuffled training instances, for each mini-batch, we shall build the computation graph, and here is the key difference between N3LDG++ and other deep learning frameworks:

.. code-block:: c++

    Graph graph; // Declare the computation graph.

    using n3ldg_plus::Node;
    std::vector<Node *> outputs;
    outputs.reserve(minibatch.size());

    std::vector<std::vector<int>> answers;
    answers.reserve(minibatch.size());

    int tgt_len_sum = 0;

    // In this loop, we only build the computation graph, but do not execute any forward
    // computation.
    for (const auto &ins : minibatch) {
        // ins.src can be either of type std::vector<std::string> or std::vector<int>, i.e.,
        // it means words or word ids in a source sentence. Suppose we use std::vector<int>
        // here.
        Node *enc_emb = n3ldg_plus::embedding(graph, ins.src, model_params.embedding);

        // 0.1 means dropout.
        Node *enc = n3ldg_plus::transformerEncoder(*enc_emb, model_params.encoder, 0.1).back();

        Node *dec_emb = n3ldg_plus::embedding(graph, ins.shifted_tgt, model_params.embedding);
        Node *dec = n3ldg_plus::transformerDecoder(*enc, *dec_emb, model_params.decoder, 0.1).back();
        Node *output = n3ldg_plus::softmax(*dec, model_params.embedding.size());
        outputs.push_back(output);
        answers.push_back(ins.tgt);
        tgt_len_sum += ins.tgt.size();
    }

    // The computation graph will automatically execute forward computations above in batch.
    // Note that matrices' shapes are not aligned in the mini-batch, but N3LDG++ can properly
    // compute them in batch.
    graph.forward();

    // 1.0f means sum reduction, and pass (1.0f / tgt_len_sum) if you want average reduction.
    dtype loss = n3ldg_plus::NLLoss(outputs, model_params.embedding.size(), answers, 1.0f);

    if (iteration % 1000 == 0) {
        cout << fmt::format("ppl:{}\n", std::exp(loss / tgt_len_sum));
    }

    graph.backward();

    optimizer->step();

The above codes show that we need not merge inputs from a mini-batch into a tensor nor append paddings.

Example of Hierarchical Model
-------------------------------

In the following, we will introduce a hierarchical model to show how it is convenient to build such models using N3LDG++.

Suppose we are tacking a text summarization problem which smmarizes a given document, i.e., a sentence list into a sentence. We can first define the instance structure as follows:

.. code-block:: c++

    struct Instance {
        vector<vector<int>> src;
        vector<int> tgt;
        vector<int> shifted_tgt;
    };

Next, we want to pass *src* to a Transformer layer where every word only attends words from the same sentence. One way is to use sophisticated attention masks, but it would be too much trouble and cause a massive waste of memory. Whereas, using N3LDG++, it is straightforward to do this as follows:

.. code-block:: c++

        for (const Instance &ins : minibatch) {
            ...

            for (const vector<int> &sentence : ins.src) {
                Node *emb = n3ldg_plus::embedding(graph, sentence, model_params.embedding);

                // model_params.sentence_encoder is the parameter to encode sentences.
                Node *enc = n3ldg_plus::transformerEncoder(*emb, ins.src.size(),
                        model_params.sentence_encoder, 0.1).back();
                ...
            }

            ...
        }

Thus the operators in *transformerEncoder* will be executed in batch during the lazy execution period.

Then suppose we want to attain sentence embeddings by using *avgPool* so that we can expand the above code as follows:

.. code-block:: c++

        for (const Instance &ins : minibatch) {
            vector<Node *> sen_embs;
            sen_embs.reserve(ins.src.size());

            for (const vector<int> &sentence : ins.src) {
                Node *emb = n3ldg_plus::embedding(graph, sentence, model_params.embedding);

                // model_params.sentence_encoder is the parameter to encode sentences.
                Node *enc = n3ldg_plus::transformerEncoder(*emb, ins.src.size(),
                        model_params.sentence_encoder, 0.1).back();
                enc = n3ldg_plus::avgPool(*enc, 512); // 512 is the hidden dim.
                sen_embs.push_back(enc);
            }

            ...
        }

As expected, N3LDG++ will execute all *avgPool* in batch, regardless of different columns of the input matrices.

Finally, based on the sentence embeddings, we can build the encoder of documents. Given the relatively small number of documents, we may want to impose stronger inductive bias by using LSTM, and the completed code of building the encoder is as follows:

.. code-block:: c++

        Graph graph;
        Node *h0 = n3ldg_plus::bucket(graph, 512, 0.0f); // The initial hidden state of LSTM.

        for (const Instance &ins : minibatch) {
            vector<Node *> sen_embs;
            sen_embs.reserve(ins.src.size());

            for (const vector<int> &sentence : ins.src) {
                Node *emb = n3ldg_plus::embedding(graph, sentence, model_params.embedding);

                // model_params.sentence_encoder is the parameter to encode sentences.
                Node *enc = n3ldg_plus::transformerEncoder(*emb, ins.src.size(),
                        model_params.sentence_encoder, 0.1).back();
                enc = n3ldg_plus::avgPool(*enc, 512); // 512 is the hidden dim.
                sen_embs.push_back(enc);
            }

            vector<Node *> doc_embs = n3ldg_plus::lstm(*h0, sen_embs, model_params.para_encoder, 0.1);
            Node *enc = n3ldg_plus::concat(doc_embs);

            ... // The decoder part.
        }

Enabling CUDA
---------------
To enable CUDA, you need specify the device id (required) and pre-allocated memory in GB (optinal) at the beginning of the program as follows:

.. code-block:: c++

    // Use device 0 to train the model. Or call n3ldg_plus::cuda::initCuda(0, 10) if you want
    // to pre-allocate 10 GBs to prevent other guys from using device 0 (not recommended
    // usage).
    n3ldg_plus::cuda::initCuda(0);
