from keras.layers import Input, multiply, Lambda, concatenate, Dense, Layer, dot, Activation, Embedding, BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
from scipy import sparse
from model.losses import max_margin_loss
from math import ceil


def sp_power_method(embedding, trans_mat, dangle_nodes, damp_rate=0.995, iter=20):
    nb_embed = len(embedding)
    for i in range(iter):
        embedding = damp_rate * (trans_mat.dot(embedding) + embedding[dangle_nodes].sum(axis=0) / nb_embed)\
                    + (1 - damp_rate) / nb_embed
        # print('sp_power_method: %d/%d'%(i+1, iter))
    return embedding

def gumbel_softmax(pi, temperature):
    gumbel_softmax_arg = (K.log(pi + K.epsilon())
                          - K.log(-K.log(K.random_uniform(K.shape(pi), 0., 1.)))) / temperature
    return K.softmax(gumbel_softmax_arg, axis=-1)


class DropMask(Lambda):
    def __init__(self):
        super(DropMask, self).__init__((lambda x : x))
        self.supports_masking = True


class MixtureLayer(Layer):
    def __init__(self, embed_len, regularizer=None, constraint=None, **kwargs):
        self.embed_len = embed_len
        self.regularizer = regularizer
        self.constraint = constraint
        super(MixtureLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.embed_len, input_shape[1]),
                                 initializer='glorot_uniform',
                                 regularizer= self.regularizer,
                                 constraint=self.constraint,
                                 name = 'mixture_weight_mat')
        super(MixtureLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        if K.int_shape(x)[-1] != 1: x = K.expand_dims(x, axis=1)
        return x * self.W

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embed_len, input_shape[1])


class ScaleLayer(Layer):
    def __init__(self, initializer='one', constraint=None, **kwargs):
        self.initializer = initializer
        self.constraint = constraint
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=input_shape[1:],
                                 initializer=self.initializer,
                                 constraint=self.constraint,
                                 name = 'scale')
        super(ScaleLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return x * self.scale


class DynamicNet(object):
    def __init__(self,
                 rep_model,
                 nb_node,
                 directed=True,
                 state_dim=10,
                 state_temper=0.1,
                 run_dynamics=True,
                 mini_power_it=20,
                 optimizer=Adam(),
                 loss=max_margin_loss,
                 loss_weights = None,
                 score_regularizer=None,
                 score_constraint=None,
                 ):
        self.rep_model = rep_model
        self.nb_node = nb_node
        self.directed = directed
        self.state_dim = state_dim
        self.state_temper = state_temper
        self.run_dynamics = run_dynamics
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.score_regularizer = score_regularizer
        self.score_constraint = score_constraint
        self.mini_power_it = mini_power_it
        self.build()

    _squeeze = Lambda(lambda x: K.squeeze(x, axis=1), name='squeeze_1')
    _l1_normalize = Lambda(lambda x: x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon()), name='l1_normalize')
    _sumlayer = Lambda(lambda x : K.sum(x, axis=-1, keepdims=True), name='sum_layer')
    _concat0 = Lambda(lambda x: K.concatenate(list(x), axis=0))

    def build_score_model(self, content_emb_len):
        from_emb = Input((content_emb_len,), name='from_emb')
        to_emb = Input((content_emb_len,), name='to_embed')
        neg_to_emb = Input((content_emb_len,), name='negto_embed')

        model_inputs = [from_emb, to_emb]
        neg_model_inputs = [from_emb, neg_to_emb]
        contrastive_model_input = [from_emb, to_emb, neg_to_emb]

        concat_rep = multiply([from_emb, to_emb], name='from_to_prod')
        neg_concat_rep = multiply([from_emb, neg_to_emb], name='from_negto_prod')

        if self.run_dynamics:
            from_state_input = Input((self.state_dim,), name='from_state_emb')
            to_state_input = Input((self.state_dim,), name='to_state_emb')
            neg_to_state_input = Input((self.state_dim,), name='neg_to_state_emb')

            state_interact = Dense(self.state_dim, use_bias=False, name='state_interact')
            state_interact_input = state_interact(to_state_input)
            neg_interact_input = state_interact(neg_to_state_input)

            if not self.directed:
                model_inputs.append(from_state_input)
                neg_model_inputs.append(from_state_input)
                contrastive_model_input.append(from_state_input)

                from_interact_input = state_interact(from_state_input)
                state_interact_input = multiply([from_interact_input, state_interact_input])
                neg_interact_input = multiply([from_interact_input, neg_interact_input])

            model_inputs.append(to_state_input)
            neg_model_inputs.append(neg_to_state_input)
            contrastive_model_input.append(to_state_input)
            contrastive_model_input.append(neg_to_state_input)

            concat_rep = concatenate([state_interact_input, concat_rep])

            # probs = self._l1_normalize(probs)
            # sample_probs = Lambda(lambda x: K.in_train_phase(gumbel_softmax(x, self.state_temper), K.softmax(x / self.state_temper)))(probs)
            # mixture_layer = MixtureLayer(K.int_shape(concat_rep)[1],
            #                              regularizer=self.score_regularizer,
            #                              constraint=self.score_constraint,
            #                              name='concat_emb_score')
            # mixture_layer(concat_rep)

            state_core_layer = Dense(self.state_dim, activation='tanh',
                                     kernel_regularizer=self.score_regularizer,
                                     kernel_constraint=self.score_constraint)
            aspect_state_scores = state_core_layer(concat_rep)
            probs = Activation('softmax')(aspect_state_scores)
            # probs = self._l1_normalize(aspect_state_scores)
            selector = Lambda(lambda x: K.in_train_phase(gumbel_softmax(x, self.state_temper),
                                                         K.softmax(x/self.state_temper)))(probs)
            aspect_state_scores = Activation('relu')(multiply([selector, aspect_state_scores]))
            # aspect_param = mixture_layer(sample_probs)
            # aspect_state_scores = Activation('relu')(dot([aspect_param, concat_rep], axes=1, name='aspect_state_scores'))
            overall_aspect_state_scores = self._sumlayer(aspect_state_scores)

            neg_concat_rep = concatenate([neg_interact_input, neg_concat_rep])
            # neg_aspect_state_scores = Activation('relu')(dot([aspect_param, neg_concat_rep], axes=1, name='neg_aspect_sstate_score'))
            neg_aspect_state_scores = state_core_layer(neg_concat_rep)
            neg_aspect_state_scores = Activation('relu')(multiply([selector, neg_aspect_state_scores]))
            overall_neg_aspect_state_scores = self._sumlayer(neg_aspect_state_scores)

        overall_score = self._sumlayer(concat_rep)
        score_model = Model(inputs=model_inputs, outputs=[overall_score], name='score_model')
        neg_overall_score = score_model(neg_model_inputs)

        score_prob_model = None
        contrastive_outputs = [overall_score, neg_overall_score]
        if self.run_dynamics:
            score_prob_model = Model(inputs=model_inputs, outputs=[aspect_state_scores, selector], name='score_prob_model')
            contrastive_outputs += [overall_aspect_state_scores, overall_neg_aspect_state_scores]

        contrastive_model = Model(inputs=contrastive_model_input, outputs=contrastive_outputs, name='contrastive_score_model')

        return score_model, contrastive_model, score_prob_model

    def build_dynamic_model(self):
        embedding = K.placeholder((self.nb_node,), name='embedding')
        trans_mat = K.placeholder(ndim=2, sparse=True, name='trans_mat')
        dangle_nodes = K.placeholder(ndim=1, dtype='int32', name='dangle_nodes')
        damp_rate = K.placeholder((1,), name='damp_rate')
        telport = (1 - damp_rate) / self.nb_node
        result = K.expand_dims(embedding)
        for i in range(self.mini_power_it):
            result = damp_rate * (K.dot(trans_mat, result) +
                        K.sum(K.gather(result, dangle_nodes)) / self.nb_node) + telport
        return K.function(inputs=[embedding, trans_mat, dangle_nodes, damp_rate], outputs=[K.squeeze(result,-1)])

    def build_state_embed(self):
        init_state = np.random.rand(self.nb_node, self.state_dim)
        init_state /= np.sum(init_state, axis=0) + np.finfo(np.float32).eps
        state_embed = Embedding(self.nb_node, self.state_dim, name='state_emb', weights=[init_state], trainable=False)
        return state_embed

    def build(self):
        dropmask = DropMask()
        self.embed_score_model, contrastive_sore_mdoel, self.score_prob_model = \
            self.build_score_model(self.rep_model.output_shape[-1])

        target_input = Input((1,), name='target_input')
        neg_target_input = Input((1,), name='neg_target_input')

        rep_input = self.rep_model.inputs

        target_rep_input = [
            Input(batch_shape=K.int_shape(input), name='target_rep_input_' + str(i))
            for i, input in enumerate(rep_input)]
        neg_target_rep_input = [
            Input(batch_shape=K.int_shape(input), name='neg_target_rep_input_' + str(i))
            for i, input in enumerate(rep_input)]

        source_rep = dropmask(self._squeeze(self.rep_model.output))
        target_rep = dropmask(self._squeeze(self.rep_model(target_rep_input)))
        neg_target_rep = dropmask(self._squeeze(self.rep_model(neg_target_rep_input)))

        score_input = [source_rep, target_rep]
        triplet_score_input = [source_rep, target_rep, neg_target_rep]

        self.state_emb = None
        if self.run_dynamics:
            self.func_power = self.build_dynamic_model()
            self.state_emb = self.build_state_embed()
            from_state = self._squeeze(self.state_emb(target_input))
            target_state = self._squeeze(self.state_emb(target_input))
            neg_target_state = self._squeeze(self.state_emb(neg_target_input))

            if not self.directed:
                score_input.append(from_state)
                triplet_score_input.append(from_state)

            score_input.append(target_state)
            triplet_score_input.append(target_state)
            triplet_score_input.append(neg_target_state)

        contrastive_outputs = contrastive_sore_mdoel(triplet_score_input)
        output_scores = [concatenate(contrastive_outputs[:2], axis=-1)]
        losses = [self.loss]
        if self.run_dynamics:
            outputs = self.score_prob_model(score_input)
            self.input_prob_model = Model(inputs=rep_input + target_rep_input + [target_input],
                                          outputs=outputs, name='input_prob_model')

            output_scores.append(concatenate(contrastive_outputs[2:], axis=-1))
            losses.append(self.loss)

        self.contrastive_model = Model(inputs=rep_input + target_rep_input + neg_target_rep_input + [target_input, neg_target_input],
                                       outputs=output_scores, name='dynamicNet')
        self.contrastive_model.compile(optimizer=self.optimizer, loss=losses, loss_weights=self.loss_weights)


    def dropout_network(self, from_nodes, to_nodes, values=None, ratio=0.1):
        if ratio > 0:
            nb_edge = len(from_nodes)
            ix = np.random.permutation(nb_edge)
            ix = ix[:int(nb_edge * ratio)]
            from_nodes = np.delete(from_nodes, ix)
            to_nodes = np.delete(to_nodes, ix)
            if values:
                values = np.delete(values, ix)

        return from_nodes, to_nodes, values


    def fit_state_embedding(self, from_nodes, to_nodes, values, selector=None, damp_rate=0.995, nb_walk_iter=10):
        if selector is None: selector = values

        aspect_stat_emb = self.state_emb.get_weights()[0]
        aspect_vol = np.sum(aspect_stat_emb, axis=0, keepdims=True)
        aspect_stat_emb /= aspect_vol

        tuple_idx = (np.arange(0,len(selector)), np.argmax(selector, axis=1))
        values = sparse.coo_matrix((values[tuple_idx], tuple_idx), shape=values.shape, dtype='float32')

        damp_rate = np.asarray([damp_rate])
        for i in range(self.state_dim):
            aspect_value = values.getcol(i)
            node_idx = aspect_value.nonzero()[0]

            transmat = sparse.coo_matrix((aspect_value.data, (from_nodes[node_idx], to_nodes[node_idx])),
                                         shape=(self.nb_node, self.nb_node),
                                         dtype=np.float32)
            sum_trans = np.squeeze(np.asarray(transmat.sum(axis=0)))
            dangle_node = np.where(sum_trans == 0)[0]
            sum_trans[dangle_node] = 1
            transmat = transmat.multiply(1 / sum_trans)

            state_emb = aspect_stat_emb[:, i]
            for j in range(ceil(float(nb_walk_iter)/self.mini_power_it)):
                state_emb = self.func_power([state_emb, transmat, dangle_node, damp_rate])[0]
            aspect_stat_emb[:, i] = state_emb
            print('Power method: %d/%d'%(i+1, self.state_dim))

        # proportion = np.sum(values, 0)
        # proportion /= np.sum(proportion)
        # aspect_stat_emb *= proportion * self.state_dim
        self.state_emb.set_weights([aspect_stat_emb])
        return aspect_stat_emb


    def fit_model(self, from_to_neg_node, feat_mat, margin_overall=1, margin_con=1, batch_size=200, nb_fit_iter=5):
        nb_sample = len(from_to_neg_node[0])
        nb_batch = int(ceil(float(nb_sample) / batch_size))
        y_overall_margin = margin_overall * np.ones(batch_size, dtype=np.float32)
        y_batch = [y_overall_margin]
        if self.run_dynamics:
            y_con_margin = margin_con * np.ones(batch_size, dtype=np.float32)
            y_batch.append(y_con_margin)

        losslist = []
        for it in range(nb_fit_iter):
            loss = 0
            for i in range(nb_batch):
                start_idx = i * batch_size
                end_idx = min((i+1)*batch_size, nb_sample)
                from_nodes = from_to_neg_node[0][start_idx:end_idx]
                to_nodes = from_to_neg_node[1][start_idx:end_idx]
                neg_nodes = from_to_neg_node[2][start_idx:end_idx]

                nb_batch_spl = end_idx-start_idx
                if  nb_batch_spl != batch_size:
                    y_in = [margin_overall*np.ones(nb_batch_spl, dtype=np.float32)]
                    if self.run_dynamics:
                        y_in.append(margin_con*np.ones(nb_batch_spl, dtype=np.float32))
                else:
                    y_in = y_batch

                loss_batch = self.contrastive_model.train_on_batch(
                            x=[from_nodes, feat_mat[from_nodes], to_nodes, feat_mat[to_nodes],
                                neg_nodes, feat_mat[neg_nodes], to_nodes, neg_nodes],
                            y=y_in)
                loss += sum(loss_batch) if self.run_dynamics else loss_batch
                if i%1000 == 0:
                    print('Fitting batch: %d/%d' % (i + 1, nb_batch))

            losslist.append(loss/nb_batch)
            print('Fitting model %d/%d, Loss: %g'%(it+1, nb_fit_iter, loss))

        return losslist


    def predict_aspect_scores(self, from_nodes, to_nodes, feat_mat, batch_size=200):
        nb_sample = len(from_nodes)
        nb_batch = int(ceil(float(nb_sample) / batch_size))

        score_mat = np.zeros((nb_sample, self.state_dim), dtype=np.float32)
        probs_mat = np.zeros((nb_sample, self.state_dim), dtype=np.float32)
        for i in range(nb_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, nb_sample)
            f_node = from_nodes[start_idx:end_idx]
            t_node = to_nodes[start_idx:end_idx]
            aspect_state_scores, selectors = self.input_prob_model.predict_on_batch(
                x=[f_node, feat_mat[f_node], t_node, feat_mat[t_node], t_node])
            score_mat[start_idx:end_idx] = aspect_state_scores
            probs_mat[start_idx:end_idx] = selectors
        return score_mat, probs_mat


    def fit(self, from_to_neg_node, feat_mat, margin_overall, margin_con, edges=None, edge_scores=None,
            batch_size=200, pred_batch_size=500, damp_rate=0.995, nb_fit_iter=1, nb_walk_iter=20):

        if self.run_dynamics:
            # Stage 1
            print('Multi-aspect dynamics learning...')
            from_nodes = edges[0]
            to_nodes = edges[1]

            if edge_scores is None:
                  edge_scores, probs = self.predict_aspect_scores(from_nodes, to_nodes, feat_mat, pred_batch_size)

            self.fit_state_embedding(from_nodes, to_nodes, edge_scores, probs, damp_rate, nb_walk_iter)

        print('Tendency learning...')
        losslist = self.fit_model(from_to_neg_node, feat_mat, margin_overall, margin_con, batch_size, nb_fit_iter)

        return losslist


    def get_features(self, nodes, feat_mat, batch_size=200):
        feat_rep = self.rep_model.predict(x=[nodes, feat_mat[nodes]], batch_size=batch_size)
        # state_rep = self.state_model.predict(x=nodes, batch_size=batch_size)

        return np.squeeze(feat_rep)

