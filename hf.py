"""
Copyright (c) 2011, Yahoo! Inc.  All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of Yahoo! Inc. nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of Yahoo! Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from theano import function, shared
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from math import isnan,isinf

import theano.tensor as T
import theano
import numpy

def linear_conj_grad_r(quad_mult_fun, rhs, preconditioner, inputs, givens):
    """
    Solve by linear conjugate gradient A*x = rhs
    The matrix A is not explicitly computed, but trough the quad_mult_fun which should return:
    quad_mult_fun(v) = A*v
    """

    x = [ shared(numpy.zeros(p.get_value().shape,dtype=theano.config.floatX)) for p in rhs ]
    direction = [ shared(numpy.zeros(p.get_value().shape,dtype=theano.config.floatX)) for p in rhs ]
    old_residual_norm_squared = shared(numpy.zeros((),dtype=theano.config.floatX))

    Ax = quad_mult_fun(x)
    residual = [ b-a for a,b in zip(Ax, rhs)]

    residual_norm_squared = T.sum([ T.sum(T.sqr(r)/p) for (r,p) in zip(residual, preconditioner) ])
    curvature = T.sum( [ T.sum(a*b) for a,b in zip( quad_mult_fun(direction), direction) ])

    obj_fun = 0.5*T.sum( [T.sum( (a+b)*c )  for a,b,c in zip(residual, rhs, x)] )

    alpha = residual_norm_squared /  curvature
    update_x = function(inputs, obj_fun, updates = [(a, a+alpha*d) for (a,d) in zip(x,direction)], givens = givens)

    beta = residual_norm_squared / old_residual_norm_squared

    iteration_updates = [ (d, r/p+beta*d) for (d,r,p) in zip(direction,residual,preconditioner) ]
    iteration_updates.append( (old_residual_norm_squared, residual_norm_squared) )

    update_direction = function(inputs, updates = iteration_updates, givens = givens)

    init_updates = [ (d,r/p) for d,r,p in zip(direction, residual, preconditioner) ]
    init_updates.append( (old_residual_norm_squared, residual_norm_squared) )
    init = function(inputs, updates = init_updates, givens = givens)

    return (x, update_x, update_direction, init)

def perform_linear_conj_grad(lcg, tol=1e-4, max_iters=50, args=()):
    (x, update_x, update_direction, init) = lcg

    iters = 0
    init(*args)

    while iters < max_iters:
        obj_fun_lcg = update_x(*args)

        print '\tCG iter', iters, 'obj_fun_lcg =', obj_fun_lcg, '\r',
        if iters > 0 and 1 - old_obj_fun/obj_fun_lcg  < tol and obj_fun_lcg>0: break
        update_direction(*args)
        iters += 1

        old_obj_fun = obj_fun_lcg
        if old_obj_fun > obj_fun_lcg:
            print >> sys.stderr, 'Warning: value of the CG objective function decreased'
    print
    return obj_fun_lcg


def truncated_newton(inputs, output, costs, params, givens, maxiter, ridge,
                     precond_type, n_train_batches, *args):
    """
    Minimization by truncated Newton
    The linear system involved in the computation of the Newton step is solved with linear conjugate gradient
    The function is: inputs -> output (typically the last layer) -> costs or costs[0] (a scalar)
      The first component of inputs should be the batch index
    params is the set of parameters to be optimized
    maxiter: maximum number of steps
    ridge: initial value of the ridge
    precond_type: jacobi, martens or none
    n_train_batches: number of training to batches to cycle over
    """

    opt_cost = costs[0] if isinstance(costs,(list,tuple)) else costs  # There might be different costs of interest, but we only minimize the first one.

    def gauss_vect_mult(v):
        """
        Multiply a vector by the Gauss-Newton matrix JHJ'
          where J is the Jacobian between output and params and H is the Hessian between costs and output
          H should be diagonal and positive.
        Also add the ridge
        """
        Jv = T.Rop(output, params, v)
        HJv = T.Rop(T.grad(opt_cost,output), output, Jv)
        JHJv = T.Lop(output, params, HJv)
        if not isinstance(JHJv,list):
            JHJv = [JHJv]
        JHJv = [a+ridge*b for a,b in zip(JHJv,v)]
        return JHJv

#----------------Ridge

    rho = 1 # The ratio between the actual decrease and the predicted decrease
    ridge = shared(numpy.array(ridge,dtype=theano.config.floatX))
    ridge_update_factor = T.scalar(dtype=theano.config.floatX)
    ridge_update = function([ridge_update_factor], [], updates = [(ridge, ridge*ridge_update_factor)])

#---------------Preconditioner

    ind_block = T.iscalar()
    nblock = 100 # Preconditioner in computed in blocks. The larger nblock, the smaller the variance, but the larger the computation time.
    preconditioner = [ shared(numpy.ones(p.get_value().shape,dtype=theano.config.floatX)) for p in params ]

    def compute_preconditioner_block():
        srng = RandomStreams(seed=1234)
        r = T.sgn(srng.normal(output.shape))
        grad = T.grad(opt_cost, output)
        if precond_type == 'jacobi':
          val = T.sqrt(T.grad(T.sum(grad), output)) * r
        elif precond_type == 'martens':
          val = grad * r
        else:
          raise NotImplementedError("Invalid preconditioner specified")

        precond = [T.sqr(v) for v in T.Lop(output[ind_block::nblock], params, val[ind_block::nblock])]
        updates = [(a,a+b) for a,b in zip(preconditioner, precond)]
        return function([ind_block]+inputs, [], givens=givens, updates=updates)

    if precond_type:
      update_precond_block = compute_preconditioner_block()
    init_preconditioner = function([], [], updates = [(a,ridge*T.ones_like(a)) for a in preconditioner])

    def update_preconditioner():
       init_preconditioner()
       if precond_type:
         for i in range(nblock):
            update_precond_block(i,*the_args)
         if precond_type == 'martens':
             function([], [], updates=[(a,a**0.75) for a in preconditioner])()


#-----------------Gradient (on the full dataset)

    grhs = [ shared(numpy.zeros(p.get_value().shape,dtype=theano.config.floatX)) for p in params ]
    gparams = T.grad(opt_cost, params)
    init_gradient = function([], [], updates = [(a,T.zeros_like(a)) for a in grhs])
    update_gradient_batch = function(inputs, costs, givens=givens,
      updates=[(a,a-b/n_train_batches) for a,b in zip(grhs, gparams)])

    def update_grhs():
      init_gradient()
      costs_per_batch = []
      for i in range(n_train_batches):
        c = update_gradient_batch(i,*args)
        costs_per_batch.append(c)
      return numpy.mean(costs_per_batch,axis=0)

#---------------------Linear CG and updates

    lcg = linear_conj_grad_r(gauss_vect_mult, grhs, preconditioner, inputs, givens)
    step = lcg[0]
    norm_step = function([], ridge*T.sum([T.sum(T.sqr(a)) for a in step]))

    starting_point_lcg = function([], [], updates = [ (a,T.zeros_like(a)) for a in step ])
    update_params = function([], [], updates = [(p,p+s) for p,s in zip(params,step)])
    backtrack_params = function([], [], updates = [(p,p-s) for p,s in zip(params,step)])

    gauss_grad_mult = function(inputs, sum([T.sum(a*b) for a,b in zip(gauss_vect_mult(grhs),grhs)]), givens = givens)

#---------------------Start

    cost_values = update_grhs()
    obj_value = cost_values[0]
    costs_log = []

    for i in range(maxiter):
        current_batch = i % n_train_batches
        the_args = (current_batch,) + args

        update_preconditioner()

        if i > 0:
            delta_obj = old_obj_value - obj_value
            rho = delta_obj / newton_decr

            if rho > 0.5 and delta_obj < obj_value * 1e-5 and delta_obj > 0:
                return costs_log

            if rho < 0.25: ridge_update(2)
            if rho > 0.75: ridge_update(0.5)

            costs_log.append(cost_values)
            print 'Iter',i, 'Objective function value =', obj_value, 'other costs = ', cost_values[1:], 'rho =', rho, 'ridge =', ridge.get_value()
        else:
            print 'Iter',i, 'Objective function value =', obj_value, 'other costs = ', cost_values[1:]

        old_obj_value = obj_value
        while 1:
            newton_decr = perform_linear_conj_grad(lcg, tol=0.0, args=the_args)
            newton_decr += 0.5*norm_step()
            if newton_decr > 0 and not isnan(newton_decr):
                update_params()
                cost_values = update_grhs()
                obj_value = cost_values[0]
                gauss_grad_mult_nan = isnan(gauss_grad_mult(*the_args))
                if gauss_grad_mult_nan:
                    print 'Gauss-Newton gradient multiplication returned nan'
                if obj_value < old_obj_value and not isnan(obj_value) and not isinf(obj_value) and not gauss_grad_mult_nan: break
                backtrack_params()
                update_grhs()
            else:
                cost_values = update_grhs()
                obj_value = cost_values[0]

            starting_point_lcg()
            ridge_update(4)
            print 'Newton decrement =', newton_decr, 'Objective function value =', obj_value, 'Increasing ridge to', ridge.get_value()

    return costs_log

