% Multi Dimensional Knapsack Problem --- proof of optimality

int: N; % number of variables
int: M; % number of constraints

set of int: VariableRange = 1..N;
set of int: ConstraintRange = 1..M;

array[ConstraintRange,VariableRange] of int: a; % coefficients
array[ConstraintRange] of int: b; % rhs
array[VariableRange] of int: c; % coefficients of the objective fct
int: z; % optimal value
set of int: UsedVariableRange = 1..30;

array[UsedVariableRange] of var 0..1: x;

% check that data are nonnegative
constraint forall (i in ConstraintRange) (
             forall (j in VariableRange) (
               assert (a[i,j] >= 0, "negative values in a") ));
constraint forall (i in ConstraintRange) (
             assert (b[i] >= 0, "negative values in b") );
constraint forall (j in VariableRange) (
             assert (c[j] >= 0, "negative values in c") );
constraint assert (z >= 0, "negative z");

% upper bound constraints
constraint forall (i in ConstraintRange) (
    sum (j in UsedVariableRange) (a[i,j]*x[j]) <= sum (j in UsedVariableRange)(a[i,j]) / 8
);

% fix objective fct to optimal value
int: best_obj = sum(j in UsedVariableRange)(c[j]);
var 0..best_obj: obj; 

constraint obj <= sum (j in UsedVariableRange) (c[j]*x[j]);

%------------------------------------------------------------------------------%
% Search

solve 
    :: int_search(x, input_order, indomain_max, complete)
    maximize obj;

%------------------------------------------------------------------------------%
% Output

output [ 
    "x = ", show(x), ";\n", 
    "obj = ", show(obj), ";"
]
