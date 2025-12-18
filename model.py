import jax
import jax.numpy as np
from memo import memo, domain 
import enum

# Define the domains for key variables
H = np.array([0, 2])     # number of hours spent working on assignment
T = np.array([1, 3])     # expected class average of time spent on assignment (low/high)
D = np.array([0, 1])     # assignment was difficult? (1 = difficult) maybe also a enum?

class S(enum.IntEnum): FLUENT = 0; STUCK = 1           # (S)tuck on the assignment?
class R(enum.IntEnum): CONFIDENT = 0; INSECURE = 1  # Insecure/Confident student?
class F(enum.IntEnum): BOTH = 0; EITHER = 1         # (F)orm of causal graph

U = domain(t=len(T) + 1, d=len(D) + 1)  # utterances; 2 = say nothing 

@jax.jit
def p_stuck(f, h, t, d, p_both=0.15, p_either=0.15):
    ps = np.array([
        [[0.01, p_both ** 2],   # BOTH
         [p_both ** 2, p_both]],
        [[0.01, p_either ** 2], # EITHER
         [p_either ** 2, p_either]],
    ])
    return ps[f, d, 1 * (h < t)]

@jax.jit
def is_true(t, d, u):
    '''Is utterance u consistent with t (threshold) and d (difficulty)?'''
    return (
        ((U.t(u) == 2) | (T[U.t(u)] == t)) &
        ((U.d(u) == 2) | (D[U.d(u)] == d))
    )

@jax.jit
def is_true_t(t, u):
    return (U.t(u) == 2) | (T[U.t(u)] == t)

@jax.jit
def is_true_d(d, u):
    # Consistency in difficulty dimension
    return (U.d(u) == 2) | (D[U.d(u)] == d)

@jax.jit
def is_not_blank(u):
    return (U.t(u) != 2) | (U.d(u) != 2)

# whether or not specifies the t dimension in explanation 
@jax.jit
def targets_hours(u):
    return 1 * (U.t(u) != 2)

#P(Stuck | u)
@memo
def student_understanding[f: F, h: H, u: U](tc, tca, p_both, p_either):
    student: knows(f, h, u)
    student: thinks[
        teacher: knows(f, h, u),
        teacher: chooses(t in T, wpp=t if t == is_true_t(t, u) else exp(tca) if t == T[0] else 1),
        teacher: chooses(d in D, wpp=d if d == is_true_d(d, u) else exp(tc) if d == 1 else 1),
    ]
    return student[ E[p_stuck(f, h, teacher.t, teacher.d, p_both, p_either)] ]

# P(T, D | u)
@memo
def student_inference[f: F, h: H, u: U, s: S, t: T, d: D](tc, tca, p_both, p_either):
    student: knows(f, h)
    student: thinks[
        teacher: knows(f, h),
        teacher: chooses(t in T, wpp=exp(tca) if t == T[0] else 1),
        teacher: chooses(d in D, wpp=exp(tc) if d == 1 else 1),
        teacher: chooses(s in S, wpp=p_stuck(f, h, t, d, p_both, p_either) if s == 1 else 1 - p_stuck(f, h, t, d, p_both, p_either)),
        teacher: chooses(u in U, wpp=is_true(t, d, u))
    ]
    student: observes [teacher.s] is s
    student: observes [teacher.u] is u
    student: knows(t, d)
    return student[ Pr[ teacher.t == t ] ] * student[ Pr[ teacher.d == d ] ]

@jax.jit
def targets_difficulty(u):
    return 1 * (U.d(u) != 2)

# if h > t
@memo
def affect_self_efficacy[f: F, r: R, h: H, u: U, s: S](tc, tca, p_both, p_either):
    student: knows(f, r, h, u)
    student: thinks[
        teacher: knows(f, h),
        teacher: chooses(t in T, wpp=exp(tca) if t == T[0] else 1),
        teacher: chooses(d in D, wpp=exp(tc) if d == 1 else 1),
        teacher: chooses(d_ in D, wpp=exp(tc) if d_== 1 else 1),
        teacher: chooses(s in S, wpp=p_stuck(f, h, t, d, p_both, p_either) if s == 1 else 1 - p_stuck(f, h, t, d, p_both, p_either)),
        teacher: chooses(u in U, wpp=is_true(t, d, u))
    ]
    student: observes [teacher.s] is s
    student: observes [teacher.u] is u
    return student[
        E[ p_stuck(f, h, teacher.t, teacher.d_, p_both, p_either) -
           p_stuck(f, 0, teacher.t, teacher.d_, p_both, p_either) ]
    ] 


@jax.jit
def targets_hours(u): 
    return 1 * (U.t(u) != 2)

# if h < t
@memo
def affect_shame[f: F, r: R, h: H, u: U, s: S](tc, tca, p_both, p_either):
    student: knows(f, r, h, u)
    student: thinks[
        teacher: knows(f, h),
        teacher: chooses(t in T, wpp=exp(tca) if t == T[0] else 1),
        teacher: chooses(d in D, wpp=exp(tc) if d == 1 else 1),
        teacher: chooses(d_ in D, wpp=exp(tc) if d_== 1 else 1),
        teacher: chooses(s in S, wpp=p_stuck(f, h, t, d, p_both, p_either) if s == 1 else 1 - p_stuck(f, h, t, d, p_both, p_either)),
        teacher: chooses(u in U, wpp=is_true(t, d, u))
    ]
    student: observes [teacher.s] is s
    student: observes [teacher.u] is u
    return student[
        E[ p_stuck(f, h, teacher.t, teacher.d_, p_both, p_either) -
           p_stuck(f, 2, teacher.t, teacher.d_, p_both, p_either) ]
    ] 

  
@memo
def teacher[f: F, r: R, h: H, t: T, d: D, u: U, s: S](
    uc, urc, tc, tca, sc, scc, p_both, p_either, alphac, alpha
):
    cast: [teacher, student, world]
    teacher: knows(f, r, h, t, d, s) 
    teacher: chooses(u in U, wpp = is_true(t, d, u) * exp(0
          + uc * student_understanding[f, h, u](tc, tca, p_both, p_either)
          + urc * student_inference[f, h, u, s, t, d](tc, tca, p_both, p_either)
          - ((scc if r == 0 else sc) + 
                ((alphac if r == 0 else alpha))) * affect_self_efficacy[f, r, h, u, s](tc, tca, p_both, p_either)
            - ((scc if r == 0 else sc) + (1 - (alphac if r == 0 else alpha))) * affect_shame[f, r, h, u, s](tc, tca, p_both, p_either)
            
    ))
    return Pr[teacher.u == u]


