??
?4?3
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
s
	AssignSub
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
;
Elu
features"T
activations"T"
Ttype:
2
L
EluGrad
	gradients"T
outputs"T
	backprops"T"
Ttype:
2
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
)
Rank

input"T

output"	
Ttype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?

ScatterAdd
ref"T?
indices"Tindices
updates"T

output_ref"T?" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
?
ScatterNdUpdate
ref"T?
indices"Tindices
updates"T

output_ref"T?"	
Ttype"
Tindicestype:
2	"
use_lockingbool(
?
ScatterUpdate
ref"T?
indices"Tindices
updates"T

output_ref"T?"	
Ttype"
Tindicestype:
2	"
use_lockingbool(
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
?
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b6??
f
PlaceholderPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
^
PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
?
PlaceholderWithDefaultPlaceholderWithDefaultPlaceholderWithDefault/input*
_output_shapes
: *
dtype0
*
shape: 
h
Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
h
Placeholder_2Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
3linear_user_feat/Initializer/truncated_normal/shapeConst*#
_class
loc:@linear_user_feat*
_output_shapes
:*
dtype0*
valueB"?A     
?
2linear_user_feat/Initializer/truncated_normal/meanConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
4linear_user_feat/Initializer/truncated_normal/stddevConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
=linear_user_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3linear_user_feat/Initializer/truncated_normal/shape*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
dtype0*

seed**
seed2 
?
1linear_user_feat/Initializer/truncated_normal/mulMul=linear_user_feat/Initializer/truncated_normal/TruncatedNormal4linear_user_feat/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
-linear_user_feat/Initializer/truncated_normalAddV21linear_user_feat/Initializer/truncated_normal/mul2linear_user_feat/Initializer/truncated_normal/mean*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
linear_user_feat
VariableV2*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
	container *
dtype0*
shape:
Ã*
shared_name 
?
linear_user_feat/AssignAssignlinear_user_feat-linear_user_feat/Initializer/truncated_normal*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
linear_user_feat/readIdentitylinear_user_feat*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
3linear_item_feat/Initializer/truncated_normal/shapeConst*#
_class
loc:@linear_item_feat*
_output_shapes
:*
dtype0*
valueB"hC     
?
2linear_item_feat/Initializer/truncated_normal/meanConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
4linear_item_feat/Initializer/truncated_normal/stddevConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
=linear_item_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3linear_item_feat/Initializer/truncated_normal/shape*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
dtype0*

seed**
seed2
?
1linear_item_feat/Initializer/truncated_normal/mulMul=linear_item_feat/Initializer/truncated_normal/TruncatedNormal4linear_item_feat/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
-linear_item_feat/Initializer/truncated_normalAddV21linear_item_feat/Initializer/truncated_normal/mul2linear_item_feat/Initializer/truncated_normal/mean*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
linear_item_feat
VariableV2*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
linear_item_feat/AssignAssignlinear_item_feat-linear_item_feat/Initializer/truncated_normal*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
linear_item_feat/readIdentitylinear_item_feat*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
2embed_user_feat/Initializer/truncated_normal/shapeConst*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0*
valueB"?A     
?
1embed_user_feat/Initializer/truncated_normal/meanConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
3embed_user_feat/Initializer/truncated_normal/stddevConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
<embed_user_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2embed_user_feat/Initializer/truncated_normal/shape*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
dtype0*

seed**
seed2
?
0embed_user_feat/Initializer/truncated_normal/mulMul<embed_user_feat/Initializer/truncated_normal/TruncatedNormal3embed_user_feat/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
,embed_user_feat/Initializer/truncated_normalAddV20embed_user_feat/Initializer/truncated_normal/mul1embed_user_feat/Initializer/truncated_normal/mean*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
embed_user_feat
VariableV2*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
	container *
dtype0*
shape:
Ã*
shared_name 
?
embed_user_feat/AssignAssignembed_user_feat,embed_user_feat/Initializer/truncated_normal*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
embed_user_feat/readIdentityembed_user_feat*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
2embed_item_feat/Initializer/truncated_normal/shapeConst*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0*
valueB"hC     
?
1embed_item_feat/Initializer/truncated_normal/meanConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
3embed_item_feat/Initializer/truncated_normal/stddevConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
<embed_item_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2embed_item_feat/Initializer/truncated_normal/shape*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
dtype0*

seed**
seed2
?
0embed_item_feat/Initializer/truncated_normal/mulMul<embed_item_feat/Initializer/truncated_normal/TruncatedNormal3embed_item_feat/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
,embed_item_feat/Initializer/truncated_normalAddV20embed_item_feat/Initializer/truncated_normal/mul1embed_item_feat/Initializer/truncated_normal/mean*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
embed_item_feat
VariableV2*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
embed_item_feat/AssignAssignembed_item_feat,embed_item_feat/Initializer/truncated_normal*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
embed_item_feat/readIdentityembed_item_feat*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
|
embedding_lookup/axisConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookupGatherV2linear_user_feat/readPlaceholder_1embedding_lookup/axis*
Taxis0*
Tindices0*
Tparams0*#
_class
loc:@linear_user_feat*'
_output_shapes
:?????????*

batch_dims 
i
embedding_lookup/IdentityIdentityembedding_lookup*
T0*'
_output_shapes
:?????????
~
embedding_lookup_1/axisConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookup_1GatherV2linear_item_feat/readPlaceholder_2embedding_lookup_1/axis*
Taxis0*
Tindices0*
Tparams0*#
_class
loc:@linear_item_feat*'
_output_shapes
:?????????*

batch_dims 
m
embedding_lookup_1/IdentityIdentityembedding_lookup_1*
T0*'
_output_shapes
:?????????
}
embedding_lookup_2/axisConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookup_2GatherV2embed_user_feat/readPlaceholder_1embedding_lookup_2/axis*
Taxis0*
Tindices0*
Tparams0*"
_class
loc:@embed_user_feat*'
_output_shapes
:?????????*

batch_dims 
m
embedding_lookup_2/IdentityIdentityembedding_lookup_2*
T0*'
_output_shapes
:?????????
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
?

ExpandDims
ExpandDimsembedding_lookup_2/IdentityExpandDims/dim*
T0*

Tdim0*+
_output_shapes
:?????????
}
embedding_lookup_3/axisConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookup_3GatherV2embed_item_feat/readPlaceholder_2embedding_lookup_3/axis*
Taxis0*
Tindices0*
Tparams0*"
_class
loc:@embed_item_feat*'
_output_shapes
:?????????*

batch_dims 
m
embedding_lookup_3/IdentityIdentityembedding_lookup_3*
T0*'
_output_shapes
:?????????
R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
?
ExpandDims_1
ExpandDimsembedding_lookup_3/IdentityExpandDims_1/dim*
T0*

Tdim0*+
_output_shapes
:?????????
}
embedding_lookup_4/axisConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookup_4GatherV2embed_user_feat/readPlaceholder_1embedding_lookup_4/axis*
Taxis0*
Tindices0*
Tparams0*"
_class
loc:@embed_user_feat*'
_output_shapes
:?????????*

batch_dims 
m
embedding_lookup_4/IdentityIdentityembedding_lookup_4*
T0*'
_output_shapes
:?????????
}
embedding_lookup_5/axisConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookup_5GatherV2embed_item_feat/readPlaceholder_2embedding_lookup_5/axis*
Taxis0*
Tindices0*
Tparams0*"
_class
loc:@embed_item_feat*'
_output_shapes
:?????????*

batch_dims 
m
embedding_lookup_5/IdentityIdentityembedding_lookup_5*
T0*'
_output_shapes
:?????????
p
Placeholder_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
5linear_sparse_feat/Initializer/truncated_normal/shapeConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
:*
dtype0*
valueB:=
?
4linear_sparse_feat/Initializer/truncated_normal/meanConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
6linear_sparse_feat/Initializer/truncated_normal/stddevConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
?linear_sparse_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5linear_sparse_feat/Initializer/truncated_normal/shape*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
dtype0*

seed**
seed2
?
3linear_sparse_feat/Initializer/truncated_normal/mulMul?linear_sparse_feat/Initializer/truncated_normal/TruncatedNormal6linear_sparse_feat/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
/linear_sparse_feat/Initializer/truncated_normalAddV23linear_sparse_feat/Initializer/truncated_normal/mul4linear_sparse_feat/Initializer/truncated_normal/mean*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
linear_sparse_feat
VariableV2*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
	container *
dtype0*
shape:=*
shared_name 
?
linear_sparse_feat/AssignAssignlinear_sparse_feat/linear_sparse_feat/Initializer/truncated_normal*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking(*
validate_shape(
?
linear_sparse_feat/readIdentitylinear_sparse_feat*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
4embed_sparse_feat/Initializer/truncated_normal/shapeConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
:*
dtype0*
valueB"=      
?
3embed_sparse_feat/Initializer/truncated_normal/meanConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
5embed_sparse_feat/Initializer/truncated_normal/stddevConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
>embed_sparse_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal4embed_sparse_feat/Initializer/truncated_normal/shape*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
dtype0*

seed**
seed2
?
2embed_sparse_feat/Initializer/truncated_normal/mulMul>embed_sparse_feat/Initializer/truncated_normal/TruncatedNormal5embed_sparse_feat/Initializer/truncated_normal/stddev*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
.embed_sparse_feat/Initializer/truncated_normalAddV22embed_sparse_feat/Initializer/truncated_normal/mul3embed_sparse_feat/Initializer/truncated_normal/mean*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
embed_sparse_feat
VariableV2*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
	container *
dtype0*
shape
:=*
shared_name 
?
embed_sparse_feat/AssignAssignembed_sparse_feat.embed_sparse_feat/Initializer/truncated_normal*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking(*
validate_shape(
?
embed_sparse_feat/readIdentityembed_sparse_feat*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
embedding_lookup_6/axisConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookup_6GatherV2linear_sparse_feat/readPlaceholder_3embedding_lookup_6/axis*
Taxis0*
Tindices0*
Tparams0*%
_class
loc:@linear_sparse_feat*'
_output_shapes
:?????????*

batch_dims 
m
embedding_lookup_6/IdentityIdentityembedding_lookup_6*
T0*'
_output_shapes
:?????????

embedding_lookup_7/axisConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *
dtype0*
value	B : 
?
embedding_lookup_7GatherV2embed_sparse_feat/readPlaceholder_3embedding_lookup_7/axis*
Taxis0*
Tindices0*
Tparams0*$
_class
loc:@embed_sparse_feat*+
_output_shapes
:?????????*

batch_dims 
q
embedding_lookup_7/IdentityIdentityembedding_lookup_7*
T0*+
_output_shapes
:?????????
^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
~
ReshapeReshapeembedding_lookup_7/IdentityReshape/shape*
T0*
Tshape0*'
_output_shapes
:?????????
p
Placeholder_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      
x
	Reshape_1ReshapePlaceholder_4Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:?????????
R
ShapeShapePlaceholder_4*
T0*
_output_shapes
:*
out_type0
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
4linear_dense_feat/Initializer/truncated_normal/shapeConst*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
dtype0*
valueB:
?
3linear_dense_feat/Initializer/truncated_normal/meanConst*$
_class
loc:@linear_dense_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
5linear_dense_feat/Initializer/truncated_normal/stddevConst*$
_class
loc:@linear_dense_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
>linear_dense_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal4linear_dense_feat/Initializer/truncated_normal/shape*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
dtype0*

seed**
seed2
?
2linear_dense_feat/Initializer/truncated_normal/mulMul>linear_dense_feat/Initializer/truncated_normal/TruncatedNormal5linear_dense_feat/Initializer/truncated_normal/stddev*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:
?
.linear_dense_feat/Initializer/truncated_normalAddV22linear_dense_feat/Initializer/truncated_normal/mul3linear_dense_feat/Initializer/truncated_normal/mean*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:
?
linear_dense_feat
VariableV2*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
linear_dense_feat/AssignAssignlinear_dense_feat.linear_dense_feat/Initializer/truncated_normal*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
use_locking(*
validate_shape(
?
linear_dense_feat/readIdentitylinear_dense_feat*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:
?
3embed_dense_feat/Initializer/truncated_normal/shapeConst*#
_class
loc:@embed_dense_feat*
_output_shapes
:*
dtype0*
valueB"      
?
2embed_dense_feat/Initializer/truncated_normal/meanConst*#
_class
loc:@embed_dense_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
4embed_dense_feat/Initializer/truncated_normal/stddevConst*#
_class
loc:@embed_dense_feat*
_output_shapes
: *
dtype0*
valueB
 *
?#<
?
=embed_dense_feat/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3embed_dense_feat/Initializer/truncated_normal/shape*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
dtype0*

seed**
seed2
?
1embed_dense_feat/Initializer/truncated_normal/mulMul=embed_dense_feat/Initializer/truncated_normal/TruncatedNormal4embed_dense_feat/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:
?
-embed_dense_feat/Initializer/truncated_normalAddV21embed_dense_feat/Initializer/truncated_normal/mul2embed_dense_feat/Initializer/truncated_normal/mean*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:
?
embed_dense_feat
VariableV2*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
	container *
dtype0*
shape
:*
shared_name 
?
embed_dense_feat/AssignAssignembed_dense_feat-embed_dense_feat/Initializer/truncated_normal*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
use_locking(*
validate_shape(
?
embed_dense_feat/readIdentityembed_dense_feat*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:
_
Tile/multiplesPackstrided_slice*
N*
T0*
_output_shapes
:*

axis 
t
TileTilelinear_dense_feat/readTile/multiples*
T0*

Tmultiples0*#
_output_shapes
:?????????
`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
k
	Reshape_2ReshapeTileReshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?????????
V
MulMul	Reshape_2Placeholder_4*
T0*'
_output_shapes
:?????????
R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
|
ExpandDims_2
ExpandDimsembed_dense_feat/readExpandDims_2/dim*
T0*

Tdim0*"
_output_shapes
:
T
Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
T
Tile_1/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :
?
Tile_1/multiplesPackstrided_sliceTile_1/multiples/1Tile_1/multiples/2*
N*
T0*
_output_shapes
:*

axis 
v
Tile_1TileExpandDims_2Tile_1/multiples*
T0*

Tmultiples0*+
_output_shapes
:?????????
U
Mul_1MulTile_1	Reshape_1*
T0*+
_output_shapes
:?????????
`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    
l
	Reshape_3ReshapeMul_1Reshape_3/shape*
T0*
Tshape0*'
_output_shapes
:????????? 
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concatConcatV2embedding_lookup/Identityembedding_lookup_1/Identityembedding_lookup_6/IdentityMulconcat/axis*
N*
T0*

Tidx0*'
_output_shapes
:?????????
O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concat_1ConcatV2
ExpandDimsExpandDims_1embedding_lookup_7/IdentityMul_1concat_1/axis*
N*
T0*

Tidx0*+
_output_shapes
:?????????
O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concat_2ConcatV2embedding_lookup_4/Identityembedding_lookup_5/IdentityReshape	Reshape_3concat_2/axis*
N*
T0*

Tidx0*'
_output_shapes
:?????????P
?
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
dtype0*

seed**
seed2
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
'dense/kernel/Initializer/random_uniformAddV2+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
dense/kernel
VariableV2*
_class
loc:@dense/kernel*
_output_shapes

:*
	container *
dtype0*
shape
:*
shared_name 
?
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
u
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*    
?

dense/bias
VariableV2*
_class
loc:@dense/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:
?
dense/MatMulMatMulconcatdense/kernel/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
?
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
z
SumSumconcat_1Sum/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims( 
G
SquareSquareSum*
T0*'
_output_shapes
:?????????
R
Square_1Squareconcat_1*
T0*+
_output_shapes
:?????????
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
~
Sum_1SumSquare_1Sum_1/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims( 
K
SubSubSquareSum_1*
T0*'
_output_shapes
:?????????
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
L
mul_2Mulmul_2/xSub*
T0*'
_output_shapes
:?????????
?
6mlp/mlp_layer1/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:*
dtype0*
valueB"P      
?
4mlp/mlp_layer1/kernel/Initializer/random_uniform/minConst*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
: *
dtype0*
valueB
 *w??
?
4mlp/mlp_layer1/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
: *
dtype0*
valueB
 *w?>
?
>mlp/mlp_layer1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6mlp/mlp_layer1/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
dtype0*

seed**
seed2	
?
4mlp/mlp_layer1/kernel/Initializer/random_uniform/subSub4mlp/mlp_layer1/kernel/Initializer/random_uniform/max4mlp/mlp_layer1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
: 
?
4mlp/mlp_layer1/kernel/Initializer/random_uniform/mulMul>mlp/mlp_layer1/kernel/Initializer/random_uniform/RandomUniform4mlp/mlp_layer1/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?
?
0mlp/mlp_layer1/kernel/Initializer/random_uniformAddV24mlp/mlp_layer1/kernel/Initializer/random_uniform/mul4mlp/mlp_layer1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?
?
mlp/mlp_layer1/kernel
VariableV2*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
	container *
dtype0*
shape:	P?*
shared_name 
?
mlp/mlp_layer1/kernel/AssignAssignmlp/mlp_layer1/kernel0mlp/mlp_layer1/kernel/Initializer/random_uniform*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
use_locking(*
validate_shape(
?
mlp/mlp_layer1/kernel/readIdentitymlp/mlp_layer1/kernel*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?
?
%mlp/mlp_layer1/bias/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer1/bias
VariableV2*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
mlp/mlp_layer1/bias/AssignAssignmlp/mlp_layer1/bias%mlp/mlp_layer1/bias/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer1/bias/readIdentitymlp/mlp_layer1/bias*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?
?
mlp/mlp_layer1/MatMulMatMulconcat_2mlp/mlp_layer1/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
mlp/mlp_layer1/BiasAddBiasAddmlp/mlp_layer1/MatMulmlp/mlp_layer1/bias/read*
T0*(
_output_shapes
:??????????*
data_formatNHWC
Y
mlp/EluElumlp/mlp_layer1/BiasAdd*
T0*(
_output_shapes
:??????????
t
mlp/dropout/cond/SwitchSwitchPlaceholderWithDefaultPlaceholderWithDefault*
T0
*
_output_shapes
: : 
a
mlp/dropout/cond/switch_tIdentitymlp/dropout/cond/Switch:1*
T0
*
_output_shapes
: 
_
mlp/dropout/cond/switch_fIdentitymlp/dropout/cond/Switch*
T0
*
_output_shapes
: 
]
mlp/dropout/cond/pred_idIdentityPlaceholderWithDefault*
T0
*
_output_shapes
: 

mlp/dropout/cond/dropout/ConstConst^mlp/dropout/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   @
?
mlp/dropout/cond/dropout/MulMul%mlp/dropout/cond/dropout/Mul/Switch:1mlp/dropout/cond/dropout/Const*
T0*(
_output_shapes
:??????????
?
#mlp/dropout/cond/dropout/Mul/SwitchSwitchmlp/Elumlp/dropout/cond/pred_id*
T0*
_class
loc:@mlp/Elu*<
_output_shapes*
(:??????????:??????????
?
mlp/dropout/cond/dropout/ShapeShape%mlp/dropout/cond/dropout/Mul/Switch:1*
T0*
_output_shapes
:*
out_type0
?
5mlp/dropout/cond/dropout/random_uniform/RandomUniformRandomUniformmlp/dropout/cond/dropout/Shape*
T0*(
_output_shapes
:??????????*
dtype0*

seed**
seed2

?
'mlp/dropout/cond/dropout/GreaterEqual/yConst^mlp/dropout/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
?
%mlp/dropout/cond/dropout/GreaterEqualGreaterEqual5mlp/dropout/cond/dropout/random_uniform/RandomUniform'mlp/dropout/cond/dropout/GreaterEqual/y*
T0*(
_output_shapes
:??????????
?
mlp/dropout/cond/dropout/CastCast%mlp/dropout/cond/dropout/GreaterEqual*

DstT0*

SrcT0
*
Truncate( *(
_output_shapes
:??????????
?
mlp/dropout/cond/dropout/Mul_1Mulmlp/dropout/cond/dropout/Mulmlp/dropout/cond/dropout/Cast*
T0*(
_output_shapes
:??????????
z
mlp/dropout/cond/IdentityIdentity mlp/dropout/cond/Identity/Switch*
T0*(
_output_shapes
:??????????
?
 mlp/dropout/cond/Identity/SwitchSwitchmlp/Elumlp/dropout/cond/pred_id*
T0*
_class
loc:@mlp/Elu*<
_output_shapes*
(:??????????:??????????
?
mlp/dropout/cond/MergeMergemlp/dropout/cond/Identitymlp/dropout/cond/dropout/Mul_1*
N*
T0**
_output_shapes
:??????????: 
?
6mlp/mlp_layer2/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
4mlp/mlp_layer2/kernel/Initializer/random_uniform/minConst*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
: *
dtype0*
valueB
 *׳ݽ
?
4mlp/mlp_layer2/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
: *
dtype0*
valueB
 *׳?=
?
>mlp/mlp_layer2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6mlp/mlp_layer2/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
dtype0*

seed**
seed2
?
4mlp/mlp_layer2/kernel/Initializer/random_uniform/subSub4mlp/mlp_layer2/kernel/Initializer/random_uniform/max4mlp/mlp_layer2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
: 
?
4mlp/mlp_layer2/kernel/Initializer/random_uniform/mulMul>mlp/mlp_layer2/kernel/Initializer/random_uniform/RandomUniform4mlp/mlp_layer2/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??
?
0mlp/mlp_layer2/kernel/Initializer/random_uniformAddV24mlp/mlp_layer2/kernel/Initializer/random_uniform/mul4mlp/mlp_layer2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??
?
mlp/mlp_layer2/kernel
VariableV2*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
mlp/mlp_layer2/kernel/AssignAssignmlp/mlp_layer2/kernel0mlp/mlp_layer2/kernel/Initializer/random_uniform*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
mlp/mlp_layer2/kernel/readIdentitymlp/mlp_layer2/kernel*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??
?
%mlp/mlp_layer2/bias/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer2/bias
VariableV2*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
mlp/mlp_layer2/bias/AssignAssignmlp/mlp_layer2/bias%mlp/mlp_layer2/bias/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer2/bias/readIdentitymlp/mlp_layer2/bias*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?
?
mlp/mlp_layer2/MatMulMatMulmlp/dropout/cond/Mergemlp/mlp_layer2/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
mlp/mlp_layer2/BiasAddBiasAddmlp/mlp_layer2/MatMulmlp/mlp_layer2/bias/read*
T0*(
_output_shapes
:??????????*
data_formatNHWC
[
	mlp/Elu_1Elumlp/mlp_layer2/BiasAdd*
T0*(
_output_shapes
:??????????
v
mlp/dropout_1/cond/SwitchSwitchPlaceholderWithDefaultPlaceholderWithDefault*
T0
*
_output_shapes
: : 
e
mlp/dropout_1/cond/switch_tIdentitymlp/dropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
c
mlp/dropout_1/cond/switch_fIdentitymlp/dropout_1/cond/Switch*
T0
*
_output_shapes
: 
_
mlp/dropout_1/cond/pred_idIdentityPlaceholderWithDefault*
T0
*
_output_shapes
: 
?
 mlp/dropout_1/cond/dropout/ConstConst^mlp/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   @
?
mlp/dropout_1/cond/dropout/MulMul'mlp/dropout_1/cond/dropout/Mul/Switch:1 mlp/dropout_1/cond/dropout/Const*
T0*(
_output_shapes
:??????????
?
%mlp/dropout_1/cond/dropout/Mul/SwitchSwitch	mlp/Elu_1mlp/dropout_1/cond/pred_id*
T0*
_class
loc:@mlp/Elu_1*<
_output_shapes*
(:??????????:??????????
?
 mlp/dropout_1/cond/dropout/ShapeShape'mlp/dropout_1/cond/dropout/Mul/Switch:1*
T0*
_output_shapes
:*
out_type0
?
7mlp/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform mlp/dropout_1/cond/dropout/Shape*
T0*(
_output_shapes
:??????????*
dtype0*

seed**
seed2
?
)mlp/dropout_1/cond/dropout/GreaterEqual/yConst^mlp/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
?
'mlp/dropout_1/cond/dropout/GreaterEqualGreaterEqual7mlp/dropout_1/cond/dropout/random_uniform/RandomUniform)mlp/dropout_1/cond/dropout/GreaterEqual/y*
T0*(
_output_shapes
:??????????
?
mlp/dropout_1/cond/dropout/CastCast'mlp/dropout_1/cond/dropout/GreaterEqual*

DstT0*

SrcT0
*
Truncate( *(
_output_shapes
:??????????
?
 mlp/dropout_1/cond/dropout/Mul_1Mulmlp/dropout_1/cond/dropout/Mulmlp/dropout_1/cond/dropout/Cast*
T0*(
_output_shapes
:??????????
~
mlp/dropout_1/cond/IdentityIdentity"mlp/dropout_1/cond/Identity/Switch*
T0*(
_output_shapes
:??????????
?
"mlp/dropout_1/cond/Identity/SwitchSwitch	mlp/Elu_1mlp/dropout_1/cond/pred_id*
T0*
_class
loc:@mlp/Elu_1*<
_output_shapes*
(:??????????:??????????
?
mlp/dropout_1/cond/MergeMergemlp/dropout_1/cond/Identity mlp/dropout_1/cond/dropout/Mul_1*
N*
T0**
_output_shapes
:??????????: 
?
6mlp/mlp_layer3/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
4mlp/mlp_layer3/kernel/Initializer/random_uniform/minConst*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
: *
dtype0*
valueB
 *׳ݽ
?
4mlp/mlp_layer3/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
: *
dtype0*
valueB
 *׳?=
?
>mlp/mlp_layer3/kernel/Initializer/random_uniform/RandomUniformRandomUniform6mlp/mlp_layer3/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
dtype0*

seed**
seed2
?
4mlp/mlp_layer3/kernel/Initializer/random_uniform/subSub4mlp/mlp_layer3/kernel/Initializer/random_uniform/max4mlp/mlp_layer3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
: 
?
4mlp/mlp_layer3/kernel/Initializer/random_uniform/mulMul>mlp/mlp_layer3/kernel/Initializer/random_uniform/RandomUniform4mlp/mlp_layer3/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??
?
0mlp/mlp_layer3/kernel/Initializer/random_uniformAddV24mlp/mlp_layer3/kernel/Initializer/random_uniform/mul4mlp/mlp_layer3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??
?
mlp/mlp_layer3/kernel
VariableV2*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
mlp/mlp_layer3/kernel/AssignAssignmlp/mlp_layer3/kernel0mlp/mlp_layer3/kernel/Initializer/random_uniform*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
mlp/mlp_layer3/kernel/readIdentitymlp/mlp_layer3/kernel*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??
?
%mlp/mlp_layer3/bias/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer3/bias
VariableV2*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
mlp/mlp_layer3/bias/AssignAssignmlp/mlp_layer3/bias%mlp/mlp_layer3/bias/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer3/bias/readIdentitymlp/mlp_layer3/bias*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?
?
mlp/mlp_layer3/MatMulMatMulmlp/dropout_1/cond/Mergemlp/mlp_layer3/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
mlp/mlp_layer3/BiasAddBiasAddmlp/mlp_layer3/MatMulmlp/mlp_layer3/bias/read*
T0*(
_output_shapes
:??????????*
data_formatNHWC
[
	mlp/Elu_2Elumlp/mlp_layer3/BiasAdd*
T0*(
_output_shapes
:??????????
v
mlp/dropout_2/cond/SwitchSwitchPlaceholderWithDefaultPlaceholderWithDefault*
T0
*
_output_shapes
: : 
e
mlp/dropout_2/cond/switch_tIdentitymlp/dropout_2/cond/Switch:1*
T0
*
_output_shapes
: 
c
mlp/dropout_2/cond/switch_fIdentitymlp/dropout_2/cond/Switch*
T0
*
_output_shapes
: 
_
mlp/dropout_2/cond/pred_idIdentityPlaceholderWithDefault*
T0
*
_output_shapes
: 
?
 mlp/dropout_2/cond/dropout/ConstConst^mlp/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   @
?
mlp/dropout_2/cond/dropout/MulMul'mlp/dropout_2/cond/dropout/Mul/Switch:1 mlp/dropout_2/cond/dropout/Const*
T0*(
_output_shapes
:??????????
?
%mlp/dropout_2/cond/dropout/Mul/SwitchSwitch	mlp/Elu_2mlp/dropout_2/cond/pred_id*
T0*
_class
loc:@mlp/Elu_2*<
_output_shapes*
(:??????????:??????????
?
 mlp/dropout_2/cond/dropout/ShapeShape'mlp/dropout_2/cond/dropout/Mul/Switch:1*
T0*
_output_shapes
:*
out_type0
?
7mlp/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform mlp/dropout_2/cond/dropout/Shape*
T0*(
_output_shapes
:??????????*
dtype0*

seed**
seed2
?
)mlp/dropout_2/cond/dropout/GreaterEqual/yConst^mlp/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
?
'mlp/dropout_2/cond/dropout/GreaterEqualGreaterEqual7mlp/dropout_2/cond/dropout/random_uniform/RandomUniform)mlp/dropout_2/cond/dropout/GreaterEqual/y*
T0*(
_output_shapes
:??????????
?
mlp/dropout_2/cond/dropout/CastCast'mlp/dropout_2/cond/dropout/GreaterEqual*

DstT0*

SrcT0
*
Truncate( *(
_output_shapes
:??????????
?
 mlp/dropout_2/cond/dropout/Mul_1Mulmlp/dropout_2/cond/dropout/Mulmlp/dropout_2/cond/dropout/Cast*
T0*(
_output_shapes
:??????????
~
mlp/dropout_2/cond/IdentityIdentity"mlp/dropout_2/cond/Identity/Switch*
T0*(
_output_shapes
:??????????
?
"mlp/dropout_2/cond/Identity/SwitchSwitch	mlp/Elu_2mlp/dropout_2/cond/pred_id*
T0*
_class
loc:@mlp/Elu_2*<
_output_shapes*
(:??????????:??????????
?
mlp/dropout_2/cond/MergeMergemlp/dropout_2/cond/Identity mlp/dropout_2/cond/dropout/Mul_1*
N*
T0**
_output_shapes
:??????????: 
O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concat_3ConcatV2dense/BiasAddmul_2mlp/dropout_2/cond/Mergeconcat_3/axis*
N*
T0*

Tidx0*(
_output_shapes
:??????????
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"     
?
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *ԇ?
?
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *ԇ>
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
dtype0*

seed**
seed2
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
?
)dense_1/kernel/Initializer/random_uniformAddV2-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
?
dense_1/kernel
VariableV2*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
	container *
dtype0*
shape:	?*
shared_name 
?
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
?
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense_1/bias
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
?
dense_1/MatMulMatMulconcat_3dense_1/kernel/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
?
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC
Z
SqueezeSqueezedense_1/BiasAdd*
T0*
_output_shapes
:*
squeeze_dims
 
r
$mean_squared_error/SquaredDifferenceSquaredDifferenceSqueezePlaceholder*
T0*
_output_shapes
:
t
/mean_squared_error/assert_broadcastable/weightsConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 
?
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
T0*#
_output_shapes
:?????????*
out_type0
?
3mean_squared_error/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference*
T0*
_output_shapes
: 
u
3mean_squared_error/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : 
?
1mean_squared_error/assert_broadcastable/is_scalarEqual3mean_squared_error/assert_broadcastable/is_scalar/x4mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: *
incompatible_shape_error(
?
=mean_squared_error/assert_broadcastable/is_valid_shape/SwitchSwitch1mean_squared_error/assert_broadcastable/is_scalar1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
?
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_tIdentity?mean_squared_error/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
?
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_fIdentity=mean_squared_error/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
?
>mean_squared_error/assert_broadcastable/is_valid_shape/pred_idIdentity1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
?
?mean_squared_error/assert_broadcastable/is_valid_shape/Switch_1Switch1mean_squared_error/assert_broadcastable/is_scalar>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0
*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
_output_shapes
: : 
?
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualdmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchfmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: *
incompatible_shape_error(
?
dmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch3mean_squared_error/assert_broadcastable/values/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*F
_class<
:8loc:@mean_squared_error/assert_broadcastable/values/rank*
_output_shapes
: : 
?
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch4mean_squared_error/assert_broadcastable/weights/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/weights/rank*
_output_shapes
: : 
?
Wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityWmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
?
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
?
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
valueB :
?????????
?
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimswmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:?????????
?
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch4mean_squared_error/assert_broadcastable/values/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*2
_output_shapes 
:?????????:?????????
?
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchsmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*2
_output_shapes 
:?????????:?????????
?
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapelmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
T0*
_output_shapes
:*
out_type0
?
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
value	B :
?
kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:?????????*

index_type0
?
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
value	B :
?
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimskmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likemmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*
T0*

Tidx0*'
_output_shapes
:?????????
?
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
valueB :
?????????
?
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*

Tdim0*
_output_shapes

: 
?
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch5mean_squared_error/assert_broadcastable/weights/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
_output_shapes

: : 
?
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchumean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
_output_shapes

: : 
?
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationnmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b*
validate_indices(
?
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
_output_shapes
: *
out_type0
?
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
value	B : 
?
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualcmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xrmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: *
incompatible_shape_error(
?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*p
_classf
dbloc:@mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
?
Vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
?
<mean_squared_error/assert_broadcastable/is_valid_shape/MergeMergeVmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeAmean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
T0
*
_output_shapes
: : 
?
-mean_squared_error/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.
~
/mean_squared_error/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=
?
/mean_squared_error/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0
}
/mean_squared_error/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=
?
/mean_squared_error/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0
z
/mean_squared_error/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=
?
:mean_squared_error/assert_broadcastable/AssertGuard/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
?
<mean_squared_error/assert_broadcastable/AssertGuard/switch_tIdentity<mean_squared_error/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
?
<mean_squared_error/assert_broadcastable/AssertGuard/switch_fIdentity:mean_squared_error/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
?
;mean_squared_error/assert_broadcastable/AssertGuard/pred_idIdentity<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 

8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp=^mean_squared_error/assert_broadcastable/AssertGuard/switch_t
?
Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependencyIdentity<mean_squared_error/assert_broadcastable/AssertGuard/switch_t9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.
?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*
valueB Bweights.shape=
?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0
?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=
?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0
?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*
valueB B
is_scalar=
?
:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertAmean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchAmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*
T0
*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
?
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Switch5mean_squared_error/assert_broadcastable/weights/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*
T0*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
_output_shapes

: : 
?
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Switch4mean_squared_error/assert_broadcastable/values/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*2
_output_shapes 
:?????????:?????????
?
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3Switch1mean_squared_error/assert_broadcastable/is_scalar;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*
T0
*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
_output_shapes
: : 
?
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Identity<mean_squared_error/assert_broadcastable/AssertGuard/switch_f;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*
T0
*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
?
9mean_squared_error/assert_broadcastable/AssertGuard/MergeMergeHmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependency*
N*
T0
*
_output_shapes
: : 
?
mean_squared_error/Cast/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/Cast/x*
T0*
_output_shapes
:
X
mean_squared_error/RankRankmean_squared_error/Mul*
T0*
_output_shapes
: 
?
mean_squared_error/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B : 
?
mean_squared_error/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B :
?
mean_squared_error/rangeRangemean_squared_error/range/startmean_squared_error/Rankmean_squared_error/range/delta*

Tidx0*#
_output_shapes
:?????????
?
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/range*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
&mean_squared_error/num_present/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB
 *    
?
$mean_squared_error/num_present/EqualEqualmean_squared_error/Cast/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: *
incompatible_shape_error(
?
)mean_squared_error/num_present/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB
 *    
?
>mean_squared_error/num_present/ones_like/Shape/shape_as_tensorConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB 
?
.mean_squared_error/num_present/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
(mean_squared_error/num_present/ones_likeFill>mean_squared_error/num_present/ones_like/Shape/shape_as_tensor.mean_squared_error/num_present/ones_like/Const*
T0*
_output_shapes
: *

index_type0
?
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
?
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB 
?
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B : 
?
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*#
_output_shapes
:?????????*
out_type0
?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
: 
?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B : 
?
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: *
incompatible_shape_error(
?
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
?
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
?
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: *
incompatible_shape_error(
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*d
_classZ
XVloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : 
?
umean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityumean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
?
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
valueB :
?????????
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:?????????
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*2
_output_shapes 
:?????????:?????????
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switch?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*2
_output_shapes 
:?????????:?????????
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShape?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
T0*
_output_shapes
:*
out_type0
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
value	B :
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*'
_output_shapes
:?????????*

index_type0
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
value	B :
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*
T0*

Tidx0*'
_output_shapes
:?????????
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
valueB :
?????????
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*

Tdim0*
_output_shapes

: 
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
_output_shapes

: : 
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switch?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
_output_shapes

: : 
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b*
validate_indices(
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
_output_shapes
: *
out_type0
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
dtype0*
value	B : 
?
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: *
incompatible_shape_error(
?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*?
_class?
??loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
?
tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergewmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergetmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
T0
*
_output_shapes
: : 
?
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.
?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB Bweights.shape=
?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0
?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=
?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0
?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB B
is_scalar=
?
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
?
Ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
?
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
?
dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.
?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*
valueB Bweights.shape=
?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0
?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=
?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0
?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
dtype0*
valueB B
is_scalar=
?
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
?
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
_output_shapes

: : 
?
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*2
_output_shapes 
:?????????:?????????
?
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
?
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
?
Wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergefmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
N*
T0
*
_output_shapes
: : 
?
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*#
_output_shapes
:?????????*
out_type0
?
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*
_output_shapes
:*

index_type0
?
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:
~
#mean_squared_error/num_present/RankRank0mean_squared_error/num_present/broadcast_weights*
T0*
_output_shapes
: 
?
*mean_squared_error/num_present/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B : 
?
*mean_squared_error/num_present/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B :
?
$mean_squared_error/num_present/rangeRange*mean_squared_error/num_present/range/start#mean_squared_error/num_present/Rank*mean_squared_error/num_present/range/delta*

Tidx0*#
_output_shapes
:?????????
?
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/range*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
mean_squared_error/Rank_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B : 
?
 mean_squared_error/range_1/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B : 
?
 mean_squared_error/range_1/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
dtype0*
value	B :
?
mean_squared_error/range_1Range mean_squared_error/range_1/startmean_squared_error/Rank_1 mean_squared_error/range_1/delta*

Tidx0*
_output_shapes
: 
?
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/range_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

mean_squared_error/valueDivNoNanmean_squared_error/Sum_1mean_squared_error/num_present*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
^
gradients/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
z
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
T0*
_output_shapes
: *

index_type0
p
-gradients/mean_squared_error/value_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
r
/gradients/mean_squared_error/value_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
=gradients/mean_squared_error/value_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/mean_squared_error/value_grad/Shape/gradients/mean_squared_error/value_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
2gradients/mean_squared_error/value_grad/div_no_nanDivNoNangradients/grad_ys_0mean_squared_error/num_present*
T0*
_output_shapes
: 
?
+gradients/mean_squared_error/value_grad/SumSum2gradients/mean_squared_error/value_grad/div_no_nan=gradients/mean_squared_error/value_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
/gradients/mean_squared_error/value_grad/ReshapeReshape+gradients/mean_squared_error/value_grad/Sum-gradients/mean_squared_error/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
m
+gradients/mean_squared_error/value_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
?
4gradients/mean_squared_error/value_grad/div_no_nan_1DivNoNan+gradients/mean_squared_error/value_grad/Negmean_squared_error/num_present*
T0*
_output_shapes
: 
?
4gradients/mean_squared_error/value_grad/div_no_nan_2DivNoNan4gradients/mean_squared_error/value_grad/div_no_nan_1mean_squared_error/num_present*
T0*
_output_shapes
: 
?
+gradients/mean_squared_error/value_grad/mulMulgradients/grad_ys_04gradients/mean_squared_error/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
?
-gradients/mean_squared_error/value_grad/Sum_1Sum+gradients/mean_squared_error/value_grad/mul?gradients/mean_squared_error/value_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
1gradients/mean_squared_error/value_grad/Reshape_1Reshape-gradients/mean_squared_error/value_grad/Sum_1/gradients/mean_squared_error/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp0^gradients/mean_squared_error/value_grad/Reshape2^gradients/mean_squared_error/value_grad/Reshape_1
?
@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity/gradients/mean_squared_error/value_grad/Reshape9^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/value_grad/Reshape*
_output_shapes
: 
?
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity1gradients/mean_squared_error/value_grad/Reshape_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mean_squared_error/value_grad/Reshape_1*
_output_shapes
: 
x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
z
7gradients/mean_squared_error/Sum_1_grad/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape@gradients/mean_squared_error/value_grad/tuple/control_dependency7gradients/mean_squared_error/Sum_1_grad/Reshape/shape_1*
T0*
Tshape0*
_output_shapes
: 
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
_output_shapes
: *
dtype0*
valueB 
?
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*
T0*

Tmultiples0*
_output_shapes
: 
?
+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
T0*#
_output_shapes
:?????????*
out_type0
?
*gradients/mean_squared_error/Sum_grad/SizeSize+gradients/mean_squared_error/Sum_grad/Shape*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
_output_shapes
: *
out_type0
?
)gradients/mean_squared_error/Sum_grad/addAddV2mean_squared_error/range*gradients/mean_squared_error/Sum_grad/Size*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:?????????
?
)gradients/mean_squared_error/Sum_grad/modFloorMod)gradients/mean_squared_error/Sum_grad/add*gradients/mean_squared_error/Sum_grad/Size*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:?????????
?
-gradients/mean_squared_error/Sum_grad/Shape_1Shape)gradients/mean_squared_error/Sum_grad/mod*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
_output_shapes
:*
out_type0
?
1gradients/mean_squared_error/Sum_grad/range/startConst*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
1gradients/mean_squared_error/Sum_grad/range/deltaConst*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
+gradients/mean_squared_error/Sum_grad/rangeRange1gradients/mean_squared_error/Sum_grad/range/start*gradients/mean_squared_error/Sum_grad/Size1gradients/mean_squared_error/Sum_grad/range/delta*

Tidx0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:?????????
?
0gradients/mean_squared_error/Sum_grad/ones/ConstConst*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
*gradients/mean_squared_error/Sum_grad/onesFill-gradients/mean_squared_error/Sum_grad/Shape_10gradients/mean_squared_error/Sum_grad/ones/Const*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:?????????*

index_type0
?
3gradients/mean_squared_error/Sum_grad/DynamicStitchDynamicStitch+gradients/mean_squared_error/Sum_grad/range)gradients/mean_squared_error/Sum_grad/mod+gradients/mean_squared_error/Sum_grad/Shape*gradients/mean_squared_error/Sum_grad/ones*
N*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:?????????
?
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
?
1gradients/mean_squared_error/Sum_grad/BroadcastToBroadcastTo-gradients/mean_squared_error/Sum_grad/Reshape+gradients/mean_squared_error/Sum_grad/Shape*
T0*

Tidx0*
_output_shapes
:
?
+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
T0*#
_output_shapes
:?????????*
out_type0
?
-gradients/mean_squared_error/Mul_grad/Shape_1Shapemean_squared_error/Cast/x*
T0*
_output_shapes
: *
out_type0
?
;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/Mul_grad/Shape-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
)gradients/mean_squared_error/Mul_grad/MulMul1gradients/mean_squared_error/Sum_grad/BroadcastTomean_squared_error/Cast/x*
T0*
_output_shapes
:
?
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference1gradients/mean_squared_error/Sum_grad/BroadcastTo*
T0*
_output_shapes
:
?
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
6gradients/mean_squared_error/Mul_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/Mul_grad/Reshape0^gradients/mean_squared_error/Mul_grad/Reshape_1
?
>gradients/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/Mul_grad/Reshape7^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mean_squared_error/Mul_grad/Reshape*
_output_shapes
:
?
@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
_output_shapes
: 
?
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
_output_shapes
: *
dtype0*
valueB
 *   @
?
7gradients/mean_squared_error/SquaredDifference_grad/MulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
7gradients/mean_squared_error/SquaredDifference_grad/subSubSqueezePlaceholder?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/Mul7gradients/mean_squared_error/SquaredDifference_grad/sub*
T0*
_output_shapes
:
?
9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapeSqueeze*
T0*#
_output_shapes
:?????????*
out_type0
?
;gradients/mean_squared_error/SquaredDifference_grad/Shape_1ShapePlaceholder*
T0*
_output_shapes
:*
out_type0
?
Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*#
_output_shapes
:?????????
?
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
?
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*
_output_shapes
:
?
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*#
_output_shapes
:?????????
k
gradients/Squeeze_grad/ShapeShapedense_1/BiasAdd*
T0*
_output_shapes
:*
out_type0
?
gradients/Squeeze_grad/ReshapeReshapeLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencygradients/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Squeeze_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC
?
/gradients/dense_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/Squeeze_grad/Reshape+^gradients/dense_1/BiasAdd_grad/BiasAddGrad
?
7gradients/dense_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Squeeze_grad/Reshape0^gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Squeeze_grad/Reshape*'
_output_shapes
:?????????
?
9gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_1/BiasAdd_grad/BiasAddGrad0^gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
$gradients/dense_1/MatMul_grad/MatMulMatMul7gradients/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(
?
&gradients/dense_1/MatMul_grad/MatMul_1MatMulconcat_37gradients/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	?*
transpose_a(*
transpose_b( 
?
.gradients/dense_1/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_1/MatMul_grad/MatMul'^gradients/dense_1/MatMul_grad/MatMul_1
?
6gradients/dense_1/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_1/MatMul_grad/MatMul/^gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_1/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
8gradients/dense_1/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_1/MatMul_grad/MatMul_1/^gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	?
^
gradients/concat_3_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
u
gradients/concat_3_grad/modFloorModconcat_3/axisgradients/concat_3_grad/Rank*
T0*
_output_shapes
: 
j
gradients/concat_3_grad/ShapeShapedense/BiasAdd*
T0*
_output_shapes
:*
out_type0
?
gradients/concat_3_grad/ShapeNShapeNdense/BiasAddmul_2mlp/dropout_2/cond/Merge*
N*
T0*&
_output_shapes
:::*
out_type0
?
$gradients/concat_3_grad/ConcatOffsetConcatOffsetgradients/concat_3_grad/modgradients/concat_3_grad/ShapeN gradients/concat_3_grad/ShapeN:1 gradients/concat_3_grad/ShapeN:2*
N*&
_output_shapes
:::
?
gradients/concat_3_grad/SliceSlice6gradients/dense_1/MatMul_grad/tuple/control_dependency$gradients/concat_3_grad/ConcatOffsetgradients/concat_3_grad/ShapeN*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_3_grad/Slice_1Slice6gradients/dense_1/MatMul_grad/tuple/control_dependency&gradients/concat_3_grad/ConcatOffset:1 gradients/concat_3_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_3_grad/Slice_2Slice6gradients/dense_1/MatMul_grad/tuple/control_dependency&gradients/concat_3_grad/ConcatOffset:2 gradients/concat_3_grad/ShapeN:2*
Index0*
T0*(
_output_shapes
:??????????
?
(gradients/concat_3_grad/tuple/group_depsNoOp^gradients/concat_3_grad/Slice ^gradients/concat_3_grad/Slice_1 ^gradients/concat_3_grad/Slice_2
?
0gradients/concat_3_grad/tuple/control_dependencyIdentitygradients/concat_3_grad/Slice)^gradients/concat_3_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_3_grad/Slice*'
_output_shapes
:?????????
?
2gradients/concat_3_grad/tuple/control_dependency_1Identitygradients/concat_3_grad/Slice_1)^gradients/concat_3_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_3_grad/Slice_1*'
_output_shapes
:?????????
?
2gradients/concat_3_grad/tuple/control_dependency_2Identitygradients/concat_3_grad/Slice_2)^gradients/concat_3_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_3_grad/Slice_2*(
_output_shapes
:??????????
?
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/concat_3_grad/tuple/control_dependency*
T0*
_output_shapes
:*
data_formatNHWC
?
-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients/concat_3_grad/tuple/control_dependency)^gradients/dense/BiasAdd_grad/BiasAddGrad
?
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/concat_3_grad/tuple/control_dependency.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_3_grad/Slice*'
_output_shapes
:?????????
?
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
_
gradients/mul_2_grad/ShapeShapemul_2/x*
T0*
_output_shapes
: *
out_type0
_
gradients/mul_2_grad/Shape_1ShapeSub*
T0*
_output_shapes
:*
out_type0
?
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/mul_2_grad/MulMul2gradients/concat_3_grad/tuple/control_dependency_1Sub*
T0*'
_output_shapes
:?????????
?
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
gradients/mul_2_grad/Mul_1Mulmul_2/x2gradients/concat_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
?
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*
_output_shapes
: 
?
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*'
_output_shapes
:?????????
?
1gradients/mlp/dropout_2/cond/Merge_grad/cond_gradSwitch2gradients/concat_3_grad/tuple/control_dependency_2mlp/dropout_2/cond/pred_id*
T0*2
_class(
&$loc:@gradients/concat_3_grad/Slice_2*<
_output_shapes*
(:??????????:??????????
t
8gradients/mlp/dropout_2/cond/Merge_grad/tuple/group_depsNoOp2^gradients/mlp/dropout_2/cond/Merge_grad/cond_grad
?
@gradients/mlp/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity1gradients/mlp/dropout_2/cond/Merge_grad/cond_grad9^gradients/mlp/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_3_grad/Slice_2*(
_output_shapes
:??????????
?
Bgradients/mlp/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity3gradients/mlp/dropout_2/cond/Merge_grad/cond_grad:19^gradients/mlp/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_3_grad/Slice_2*(
_output_shapes
:??????????
?
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b(
?
$gradients/dense/MatMul_grad/MatMul_1MatMulconcat5gradients/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
?
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
?
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes

:
^
gradients/Sub_grad/ShapeShapeSquare*
T0*
_output_shapes
:*
out_type0
_
gradients/Sub_grad/Shape_1ShapeSum_1*
T0*
_output_shapes
:*
out_type0
?
(gradients/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Sub_grad/Shapegradients/Sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/Sub_grad/SumSum/gradients/mul_2_grad/tuple/control_dependency_1(gradients/Sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/Sub_grad/ReshapeReshapegradients/Sub_grad/Sumgradients/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients/Sub_grad/NegNeg/gradients/mul_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
gradients/Sub_grad/Sum_1Sumgradients/Sub_grad/Neg*gradients/Sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/Sub_grad/Reshape_1Reshapegradients/Sub_grad/Sum_1gradients/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
g
#gradients/Sub_grad/tuple/group_depsNoOp^gradients/Sub_grad/Reshape^gradients/Sub_grad/Reshape_1
?
+gradients/Sub_grad/tuple/control_dependencyIdentitygradients/Sub_grad/Reshape$^gradients/Sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Sub_grad/Reshape*'
_output_shapes
:?????????
?
-gradients/Sub_grad/tuple/control_dependency_1Identitygradients/Sub_grad/Reshape_1$^gradients/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Sub_grad/Reshape_1*'
_output_shapes
:?????????
?
5gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/ShapeShapemlp/dropout_2/cond/dropout/Mul*
T0*
_output_shapes
:*
out_type0
?
7gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Shape_1Shapemlp/dropout_2/cond/dropout/Cast*
T0*
_output_shapes
:*
out_type0
?
Egradients/mlp/dropout_2/cond/dropout/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Shape7gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
3gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/MulMulBgradients/mlp/dropout_2/cond/Merge_grad/tuple/control_dependency_1mlp/dropout_2/cond/dropout/Cast*
T0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/SumSum3gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/MulEgradients/mlp/dropout_2/cond/dropout/Mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
7gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/ReshapeReshape3gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Sum5gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
5gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Mul_1Mulmlp/dropout_2/cond/dropout/MulBgradients/mlp/dropout_2/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:??????????
?
5gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Sum_1Sum5gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Mul_1Ggradients/mlp/dropout_2/cond/dropout/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
9gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Reshape_1Reshape5gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Sum_17gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:??????????
?
@gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/tuple/group_depsNoOp8^gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Reshape:^gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Reshape_1
?
Hgradients/mlp/dropout_2/cond/dropout/Mul_1_grad/tuple/control_dependencyIdentity7gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/ReshapeA^gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Reshape*(
_output_shapes
:??????????
?
Jgradients/mlp/dropout_2/cond/dropout/Mul_1_grad/tuple/control_dependency_1Identity9gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Reshape_1A^gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/mlp/dropout_2/cond/dropout/Mul_1_grad/Reshape_1*(
_output_shapes
:??????????
\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0*
_output_shapes
: 
t
gradients/concat_grad/ShapeShapeembedding_lookup/Identity*
T0*
_output_shapes
:*
out_type0
?
gradients/concat_grad/ShapeNShapeNembedding_lookup/Identityembedding_lookup_1/Identityembedding_lookup_6/IdentityMul*
N*
T0*,
_output_shapes
::::*
out_type0
?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3*
N*,
_output_shapes
::::
?
gradients/concat_grad/SliceSlice4gradients/dense/MatMul_grad/tuple/control_dependency"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_grad/Slice_1Slice4gradients/dense/MatMul_grad/tuple/control_dependency$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_grad/Slice_2Slice4gradients/dense/MatMul_grad/tuple/control_dependency$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_grad/Slice_3Slice4gradients/dense/MatMul_grad/tuple/control_dependency$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*
Index0*
T0*'
_output_shapes
:?????????
?
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3
?
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/concat_grad/Slice*'
_output_shapes
:?????????
?
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1*'
_output_shapes
:?????????
?
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_2*'
_output_shapes
:?????????
?
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_3*'
_output_shapes
:?????????
?
gradients/Square_grad/ConstConst,^gradients/Sub_grad/tuple/control_dependency*
_output_shapes
: *
dtype0*
valueB
 *   @
t
gradients/Square_grad/MulMulSumgradients/Square_grad/Const*
T0*'
_output_shapes
:?????????
?
gradients/Square_grad/Mul_1Mul+gradients/Sub_grad/tuple/control_dependencygradients/Square_grad/Mul*
T0*'
_output_shapes
:?????????
b
gradients/Sum_1_grad/ShapeShapeSquare_1*
T0*
_output_shapes
:*
out_type0
?
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Sum_1_grad/addAddV2Sum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
?
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
?
gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
?
 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
?
gradients/Sum_1_grad/ones/ConstConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Sum_1_grad/onesFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/ones/Const*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *

index_type0
?
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/ones*
N*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
?
gradients/Sum_1_grad/ReshapeReshape-gradients/Sub_grad/tuple/control_dependency_1"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
 gradients/Sum_1_grad/BroadcastToBroadcastTogradients/Sum_1_grad/Reshapegradients/Sum_1_grad/Shape*
T0*

Tidx0*+
_output_shapes
:?????????
?
gradients/SwitchSwitch	mlp/Elu_2mlp/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:??????????:??????????
e
gradients/IdentityIdentitygradients/Switch:1*
T0*(
_output_shapes
:??????????
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
_output_shapes
:*
out_type0
o
gradients/zeros/ConstConst^gradients/Identity*
_output_shapes
: *
dtype0*
valueB
 *    
?
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:??????????*

index_type0
?
;gradients/mlp/dropout_2/cond/Identity/Switch_grad/cond_gradMerge@gradients/mlp/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros*
N*
T0**
_output_shapes
:??????????: 
?
3gradients/mlp/dropout_2/cond/dropout/Mul_grad/ShapeShape'mlp/dropout_2/cond/dropout/Mul/Switch:1*
T0*
_output_shapes
:*
out_type0
?
5gradients/mlp/dropout_2/cond/dropout/Mul_grad/Shape_1Shape mlp/dropout_2/cond/dropout/Const*
T0*
_output_shapes
: *
out_type0
?
Cgradients/mlp/dropout_2/cond/dropout/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/mlp/dropout_2/cond/dropout/Mul_grad/Shape5gradients/mlp/dropout_2/cond/dropout/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients/mlp/dropout_2/cond/dropout/Mul_grad/MulMulHgradients/mlp/dropout_2/cond/dropout/Mul_1_grad/tuple/control_dependency mlp/dropout_2/cond/dropout/Const*
T0*(
_output_shapes
:??????????
?
1gradients/mlp/dropout_2/cond/dropout/Mul_grad/SumSum1gradients/mlp/dropout_2/cond/dropout/Mul_grad/MulCgradients/mlp/dropout_2/cond/dropout/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
5gradients/mlp/dropout_2/cond/dropout/Mul_grad/ReshapeReshape1gradients/mlp/dropout_2/cond/dropout/Mul_grad/Sum3gradients/mlp/dropout_2/cond/dropout/Mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout_2/cond/dropout/Mul_grad/Mul_1Mul'mlp/dropout_2/cond/dropout/Mul/Switch:1Hgradients/mlp/dropout_2/cond/dropout/Mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout_2/cond/dropout/Mul_grad/Sum_1Sum3gradients/mlp/dropout_2/cond/dropout/Mul_grad/Mul_1Egradients/mlp/dropout_2/cond/dropout/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
7gradients/mlp/dropout_2/cond/dropout/Mul_grad/Reshape_1Reshape3gradients/mlp/dropout_2/cond/dropout/Mul_grad/Sum_15gradients/mlp/dropout_2/cond/dropout/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
>gradients/mlp/dropout_2/cond/dropout/Mul_grad/tuple/group_depsNoOp6^gradients/mlp/dropout_2/cond/dropout/Mul_grad/Reshape8^gradients/mlp/dropout_2/cond/dropout/Mul_grad/Reshape_1
?
Fgradients/mlp/dropout_2/cond/dropout/Mul_grad/tuple/control_dependencyIdentity5gradients/mlp/dropout_2/cond/dropout/Mul_grad/Reshape?^gradients/mlp/dropout_2/cond/dropout/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/mlp/dropout_2/cond/dropout/Mul_grad/Reshape*(
_output_shapes
:??????????
?
Hgradients/mlp/dropout_2/cond/dropout/Mul_grad/tuple/control_dependency_1Identity7gradients/mlp/dropout_2/cond/dropout/Mul_grad/Reshape_1?^gradients/mlp/dropout_2/cond/dropout/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mlp/dropout_2/cond/dropout/Mul_grad/Reshape_1*
_output_shapes
: 
a
gradients/Mul_grad/ShapeShape	Reshape_2*
T0*
_output_shapes
:*
out_type0
g
gradients/Mul_grad/Shape_1ShapePlaceholder_4*
T0*
_output_shapes
:*
out_type0
?
(gradients/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_grad/Shapegradients/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/Mul_grad/MulMul0gradients/concat_grad/tuple/control_dependency_3Placeholder_4*
T0*'
_output_shapes
:?????????
?
gradients/Mul_grad/SumSumgradients/Mul_grad/Mul(gradients/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/Mul_grad/ReshapeReshapegradients/Mul_grad/Sumgradients/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients/Mul_grad/Mul_1Mul	Reshape_20gradients/concat_grad/tuple/control_dependency_3*
T0*'
_output_shapes
:?????????
?
gradients/Mul_grad/Sum_1Sumgradients/Mul_grad/Mul_1*gradients/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/Mul_grad/Reshape_1Reshapegradients/Mul_grad/Sum_1gradients/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
g
#gradients/Mul_grad/tuple/group_depsNoOp^gradients/Mul_grad/Reshape^gradients/Mul_grad/Reshape_1
?
+gradients/Mul_grad/tuple/control_dependencyIdentitygradients/Mul_grad/Reshape$^gradients/Mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Mul_grad/Reshape*'
_output_shapes
:?????????
?
-gradients/Mul_grad/tuple/control_dependency_1Identitygradients/Mul_grad/Reshape_1$^gradients/Mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Mul_grad/Reshape_1*'
_output_shapes
:?????????
`
gradients/Sum_grad/ShapeShapeconcat_1*
T0*
_output_shapes
:*
out_type0
?
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Sum_grad/addAddV2Sum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
?
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
?
gradients/Sum_grad/ones/ConstConst*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Sum_grad/onesFillgradients/Sum_grad/Shape_1gradients/Sum_grad/ones/Const*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *

index_type0
?
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/ones*
N*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
?
gradients/Sum_grad/ReshapeReshapegradients/Square_grad/Mul_1 gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
gradients/Sum_grad/BroadcastToBroadcastTogradients/Sum_grad/Reshapegradients/Sum_grad/Shape*
T0*

Tidx0*+
_output_shapes
:?????????
?
gradients/Square_1_grad/ConstConst!^gradients/Sum_1_grad/BroadcastTo*
_output_shapes
: *
dtype0*
valueB
 *   @
?
gradients/Square_1_grad/MulMulconcat_1gradients/Square_1_grad/Const*
T0*+
_output_shapes
:?????????
?
gradients/Square_1_grad/Mul_1Mul gradients/Sum_1_grad/BroadcastTogradients/Square_1_grad/Mul*
T0*+
_output_shapes
:?????????
?
gradients/Switch_1Switch	mlp/Elu_2mlp/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:??????????:??????????
g
gradients/Identity_1Identitygradients/Switch_1*
T0*(
_output_shapes
:??????????
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
_output_shapes
:*
out_type0
s
gradients/zeros_1/ConstConst^gradients/Identity_1*
_output_shapes
: *
dtype0*
valueB
 *    
?
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*(
_output_shapes
:??????????*

index_type0
?
>gradients/mlp/dropout_2/cond/dropout/Mul/Switch_grad/cond_gradMergegradients/zeros_1Fgradients/mlp/dropout_2/cond/dropout/Mul_grad/tuple/control_dependency*
N*
T0**
_output_shapes
:??????????: 
?
%gradients/embedding_lookup_grad/ShapeConst*#
_class
loc:@linear_user_feat*
_output_shapes
:*
dtype0	*%
valueB	"?A             
?
$gradients/embedding_lookup_grad/CastCast%gradients/embedding_lookup_grad/Shape*

DstT0*

SrcT0	*
Truncate( *#
_class
loc:@linear_user_feat*
_output_shapes
:
l
$gradients/embedding_lookup_grad/SizeSizePlaceholder_1*
T0*
_output_shapes
: *
out_type0
p
.gradients/embedding_lookup_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
*gradients/embedding_lookup_grad/ExpandDims
ExpandDims$gradients/embedding_lookup_grad/Size.gradients/embedding_lookup_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
}
3gradients/embedding_lookup_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:

5gradients/embedding_lookup_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 

5gradients/embedding_lookup_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
-gradients/embedding_lookup_grad/strided_sliceStridedSlice$gradients/embedding_lookup_grad/Cast3gradients/embedding_lookup_grad/strided_slice/stack5gradients/embedding_lookup_grad/strided_slice/stack_15gradients/embedding_lookup_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
m
+gradients/embedding_lookup_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
&gradients/embedding_lookup_grad/concatConcatV2*gradients/embedding_lookup_grad/ExpandDims-gradients/embedding_lookup_grad/strided_slice+gradients/embedding_lookup_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
'gradients/embedding_lookup_grad/ReshapeReshape.gradients/concat_grad/tuple/control_dependency&gradients/embedding_lookup_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
)gradients/embedding_lookup_grad/Reshape_1ReshapePlaceholder_1*gradients/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
'gradients/embedding_lookup_1_grad/ShapeConst*#
_class
loc:@linear_item_feat*
_output_shapes
:*
dtype0	*%
valueB	"hC             
?
&gradients/embedding_lookup_1_grad/CastCast'gradients/embedding_lookup_1_grad/Shape*

DstT0*

SrcT0	*
Truncate( *#
_class
loc:@linear_item_feat*
_output_shapes
:
n
&gradients/embedding_lookup_1_grad/SizeSizePlaceholder_2*
T0*
_output_shapes
: *
out_type0
r
0gradients/embedding_lookup_1_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
,gradients/embedding_lookup_1_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_1_grad/Size0gradients/embedding_lookup_1_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

5gradients/embedding_lookup_1_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
7gradients/embedding_lookup_1_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
7gradients/embedding_lookup_1_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
/gradients/embedding_lookup_1_grad/strided_sliceStridedSlice&gradients/embedding_lookup_1_grad/Cast5gradients/embedding_lookup_1_grad/strided_slice/stack7gradients/embedding_lookup_1_grad/strided_slice/stack_17gradients/embedding_lookup_1_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
o
-gradients/embedding_lookup_1_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
(gradients/embedding_lookup_1_grad/concatConcatV2,gradients/embedding_lookup_1_grad/ExpandDims/gradients/embedding_lookup_1_grad/strided_slice-gradients/embedding_lookup_1_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
)gradients/embedding_lookup_1_grad/ReshapeReshape0gradients/concat_grad/tuple/control_dependency_1(gradients/embedding_lookup_1_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
+gradients/embedding_lookup_1_grad/Reshape_1ReshapePlaceholder_2,gradients/embedding_lookup_1_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
'gradients/embedding_lookup_6_grad/ShapeConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
:*
dtype0	*
valueB	R=
?
&gradients/embedding_lookup_6_grad/CastCast'gradients/embedding_lookup_6_grad/Shape*

DstT0*

SrcT0	*
Truncate( *%
_class
loc:@linear_sparse_feat*
_output_shapes
:
n
&gradients/embedding_lookup_6_grad/SizeSizePlaceholder_3*
T0*
_output_shapes
: *
out_type0
r
0gradients/embedding_lookup_6_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
,gradients/embedding_lookup_6_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_6_grad/Size0gradients/embedding_lookup_6_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

5gradients/embedding_lookup_6_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
7gradients/embedding_lookup_6_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
7gradients/embedding_lookup_6_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
/gradients/embedding_lookup_6_grad/strided_sliceStridedSlice&gradients/embedding_lookup_6_grad/Cast5gradients/embedding_lookup_6_grad/strided_slice/stack7gradients/embedding_lookup_6_grad/strided_slice/stack_17gradients/embedding_lookup_6_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
o
-gradients/embedding_lookup_6_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
(gradients/embedding_lookup_6_grad/concatConcatV2,gradients/embedding_lookup_6_grad/ExpandDims/gradients/embedding_lookup_6_grad/strided_slice-gradients/embedding_lookup_6_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
)gradients/embedding_lookup_6_grad/ReshapeReshape0gradients/concat_grad/tuple/control_dependency_2(gradients/embedding_lookup_6_grad/concat*
T0*
Tshape0*#
_output_shapes
:?????????
?
+gradients/embedding_lookup_6_grad/Reshape_1ReshapePlaceholder_3,gradients/embedding_lookup_6_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
b
gradients/Reshape_2_grad/ShapeShapeTile*
T0*
_output_shapes
:*
out_type0
?
 gradients/Reshape_2_grad/ReshapeReshape+gradients/Mul_grad/tuple/control_dependencygradients/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
gradients/AddNAddNgradients/Sum_grad/BroadcastTogradients/Square_1_grad/Mul_1*
N*
T0*1
_class'
%#loc:@gradients/Sum_grad/BroadcastTo*+
_output_shapes
:?????????
^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
u
gradients/concat_1_grad/modFloorModconcat_1/axisgradients/concat_1_grad/Rank*
T0*
_output_shapes
: 
g
gradients/concat_1_grad/ShapeShape
ExpandDims*
T0*
_output_shapes
:*
out_type0
?
gradients/concat_1_grad/ShapeNShapeN
ExpandDimsExpandDims_1embedding_lookup_7/IdentityMul_1*
N*
T0*,
_output_shapes
::::*
out_type0
?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/modgradients/concat_1_grad/ShapeN gradients/concat_1_grad/ShapeN:1 gradients/concat_1_grad/ShapeN:2 gradients/concat_1_grad/ShapeN:3*
N*,
_output_shapes
::::
?
gradients/concat_1_grad/SliceSlicegradients/AddN$gradients/concat_1_grad/ConcatOffsetgradients/concat_1_grad/ShapeN*
Index0*
T0*+
_output_shapes
:?????????
?
gradients/concat_1_grad/Slice_1Slicegradients/AddN&gradients/concat_1_grad/ConcatOffset:1 gradients/concat_1_grad/ShapeN:1*
Index0*
T0*+
_output_shapes
:?????????
?
gradients/concat_1_grad/Slice_2Slicegradients/AddN&gradients/concat_1_grad/ConcatOffset:2 gradients/concat_1_grad/ShapeN:2*
Index0*
T0*+
_output_shapes
:?????????
?
gradients/concat_1_grad/Slice_3Slicegradients/AddN&gradients/concat_1_grad/ConcatOffset:3 gradients/concat_1_grad/ShapeN:3*
Index0*
T0*+
_output_shapes
:?????????
?
(gradients/concat_1_grad/tuple/group_depsNoOp^gradients/concat_1_grad/Slice ^gradients/concat_1_grad/Slice_1 ^gradients/concat_1_grad/Slice_2 ^gradients/concat_1_grad/Slice_3
?
0gradients/concat_1_grad/tuple/control_dependencyIdentitygradients/concat_1_grad/Slice)^gradients/concat_1_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_1_grad/Slice*+
_output_shapes
:?????????
?
2gradients/concat_1_grad/tuple/control_dependency_1Identitygradients/concat_1_grad/Slice_1)^gradients/concat_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_1_grad/Slice_1*+
_output_shapes
:?????????
?
2gradients/concat_1_grad/tuple/control_dependency_2Identitygradients/concat_1_grad/Slice_2)^gradients/concat_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_1_grad/Slice_2*+
_output_shapes
:?????????
?
2gradients/concat_1_grad/tuple/control_dependency_3Identitygradients/concat_1_grad/Slice_3)^gradients/concat_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_1_grad/Slice_3*+
_output_shapes
:?????????
?
gradients/AddN_1AddN;gradients/mlp/dropout_2/cond/Identity/Switch_grad/cond_grad>gradients/mlp/dropout_2/cond/dropout/Mul/Switch_grad/cond_grad*
N*
T0*N
_classD
B@loc:@gradients/mlp/dropout_2/cond/Identity/Switch_grad/cond_grad*(
_output_shapes
:??????????
{
 gradients/mlp/Elu_2_grad/EluGradEluGradgradients/AddN_1	mlp/Elu_2*
T0*(
_output_shapes
:??????????
c
gradients/Tile_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
?
gradients/Tile_grad/stackPackTile/multiplesgradients/Tile_grad/Shape*
N*
T0*
_output_shapes

:*

axis 
s
"gradients/Tile_grad/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
?
gradients/Tile_grad/transpose	Transposegradients/Tile_grad/stack"gradients/Tile_grad/transpose/perm*
T0*
Tperm0*
_output_shapes

:
t
!gradients/Tile_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
gradients/Tile_grad/ReshapeReshapegradients/Tile_grad/transpose!gradients/Tile_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Z
gradients/Tile_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
a
gradients/Tile_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
a
gradients/Tile_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Tile_grad/rangeRangegradients/Tile_grad/range/startgradients/Tile_grad/Sizegradients/Tile_grad/range/delta*

Tidx0*
_output_shapes
:
?
gradients/Tile_grad/Reshape_1Reshape gradients/Reshape_2_grad/Reshapegradients/Tile_grad/Reshape*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
gradients/Tile_grad/SumSumgradients/Tile_grad/Reshape_1gradients/Tile_grad/range*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
z
gradients/ExpandDims_grad/ShapeShapeembedding_lookup_2/Identity*
T0*
_output_shapes
:*
out_type0
?
!gradients/ExpandDims_grad/ReshapeReshape0gradients/concat_1_grad/tuple/control_dependencygradients/ExpandDims_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
|
!gradients/ExpandDims_1_grad/ShapeShapeembedding_lookup_3/Identity*
T0*
_output_shapes
:*
out_type0
?
#gradients/ExpandDims_1_grad/ReshapeReshape2gradients/concat_1_grad/tuple/control_dependency_1!gradients/ExpandDims_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
1gradients/mlp/mlp_layer3/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/mlp/Elu_2_grad/EluGrad*
T0*
_output_shapes	
:?*
data_formatNHWC
?
6gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/group_depsNoOp!^gradients/mlp/Elu_2_grad/EluGrad2^gradients/mlp/mlp_layer3/BiasAdd_grad/BiasAddGrad
?
>gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/control_dependencyIdentity gradients/mlp/Elu_2_grad/EluGrad7^gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/mlp/Elu_2_grad/EluGrad*(
_output_shapes
:??????????
?
@gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/mlp/mlp_layer3/BiasAdd_grad/BiasAddGrad7^gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mlp/mlp_layer3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
+gradients/mlp/mlp_layer3/MatMul_grad/MatMulMatMul>gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/control_dependencymlp/mlp_layer3/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(
?
-gradients/mlp/mlp_layer3/MatMul_grad/MatMul_1MatMulmlp/dropout_1/cond/Merge>gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
??*
transpose_a(*
transpose_b( 
?
5gradients/mlp/mlp_layer3/MatMul_grad/tuple/group_depsNoOp,^gradients/mlp/mlp_layer3/MatMul_grad/MatMul.^gradients/mlp/mlp_layer3/MatMul_grad/MatMul_1
?
=gradients/mlp/mlp_layer3/MatMul_grad/tuple/control_dependencyIdentity+gradients/mlp/mlp_layer3/MatMul_grad/MatMul6^gradients/mlp/mlp_layer3/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer3/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
?gradients/mlp/mlp_layer3/MatMul_grad/tuple/control_dependency_1Identity-gradients/mlp/mlp_layer3/MatMul_grad/MatMul_16^gradients/mlp/mlp_layer3/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mlp/mlp_layer3/MatMul_grad/MatMul_1* 
_output_shapes
:
??
?
'gradients/embedding_lookup_2_grad/ShapeConst*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0	*%
valueB	"?A             
?
&gradients/embedding_lookup_2_grad/CastCast'gradients/embedding_lookup_2_grad/Shape*

DstT0*

SrcT0	*
Truncate( *"
_class
loc:@embed_user_feat*
_output_shapes
:
n
&gradients/embedding_lookup_2_grad/SizeSizePlaceholder_1*
T0*
_output_shapes
: *
out_type0
r
0gradients/embedding_lookup_2_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
,gradients/embedding_lookup_2_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_2_grad/Size0gradients/embedding_lookup_2_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

5gradients/embedding_lookup_2_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
7gradients/embedding_lookup_2_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
7gradients/embedding_lookup_2_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
/gradients/embedding_lookup_2_grad/strided_sliceStridedSlice&gradients/embedding_lookup_2_grad/Cast5gradients/embedding_lookup_2_grad/strided_slice/stack7gradients/embedding_lookup_2_grad/strided_slice/stack_17gradients/embedding_lookup_2_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
o
-gradients/embedding_lookup_2_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
(gradients/embedding_lookup_2_grad/concatConcatV2,gradients/embedding_lookup_2_grad/ExpandDims/gradients/embedding_lookup_2_grad/strided_slice-gradients/embedding_lookup_2_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
)gradients/embedding_lookup_2_grad/ReshapeReshape!gradients/ExpandDims_grad/Reshape(gradients/embedding_lookup_2_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
+gradients/embedding_lookup_2_grad/Reshape_1ReshapePlaceholder_1,gradients/embedding_lookup_2_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
'gradients/embedding_lookup_3_grad/ShapeConst*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0	*%
valueB	"hC             
?
&gradients/embedding_lookup_3_grad/CastCast'gradients/embedding_lookup_3_grad/Shape*

DstT0*

SrcT0	*
Truncate( *"
_class
loc:@embed_item_feat*
_output_shapes
:
n
&gradients/embedding_lookup_3_grad/SizeSizePlaceholder_2*
T0*
_output_shapes
: *
out_type0
r
0gradients/embedding_lookup_3_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
,gradients/embedding_lookup_3_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_3_grad/Size0gradients/embedding_lookup_3_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

5gradients/embedding_lookup_3_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
7gradients/embedding_lookup_3_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
7gradients/embedding_lookup_3_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
/gradients/embedding_lookup_3_grad/strided_sliceStridedSlice&gradients/embedding_lookup_3_grad/Cast5gradients/embedding_lookup_3_grad/strided_slice/stack7gradients/embedding_lookup_3_grad/strided_slice/stack_17gradients/embedding_lookup_3_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
o
-gradients/embedding_lookup_3_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
(gradients/embedding_lookup_3_grad/concatConcatV2,gradients/embedding_lookup_3_grad/ExpandDims/gradients/embedding_lookup_3_grad/strided_slice-gradients/embedding_lookup_3_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
)gradients/embedding_lookup_3_grad/ReshapeReshape#gradients/ExpandDims_1_grad/Reshape(gradients/embedding_lookup_3_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
+gradients/embedding_lookup_3_grad/Reshape_1ReshapePlaceholder_2,gradients/embedding_lookup_3_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
1gradients/mlp/dropout_1/cond/Merge_grad/cond_gradSwitch=gradients/mlp/mlp_layer3/MatMul_grad/tuple/control_dependencymlp/dropout_1/cond/pred_id*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer3/MatMul_grad/MatMul*<
_output_shapes*
(:??????????:??????????
t
8gradients/mlp/dropout_1/cond/Merge_grad/tuple/group_depsNoOp2^gradients/mlp/dropout_1/cond/Merge_grad/cond_grad
?
@gradients/mlp/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity1gradients/mlp/dropout_1/cond/Merge_grad/cond_grad9^gradients/mlp/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer3/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
Bgradients/mlp/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity3gradients/mlp/dropout_1/cond/Merge_grad/cond_grad:19^gradients/mlp/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer3/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
5gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/ShapeShapemlp/dropout_1/cond/dropout/Mul*
T0*
_output_shapes
:*
out_type0
?
7gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Shape_1Shapemlp/dropout_1/cond/dropout/Cast*
T0*
_output_shapes
:*
out_type0
?
Egradients/mlp/dropout_1/cond/dropout/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Shape7gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
3gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/MulMulBgradients/mlp/dropout_1/cond/Merge_grad/tuple/control_dependency_1mlp/dropout_1/cond/dropout/Cast*
T0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/SumSum3gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/MulEgradients/mlp/dropout_1/cond/dropout/Mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
7gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/ReshapeReshape3gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Sum5gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
5gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Mul_1Mulmlp/dropout_1/cond/dropout/MulBgradients/mlp/dropout_1/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:??????????
?
5gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Sum_1Sum5gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Mul_1Ggradients/mlp/dropout_1/cond/dropout/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
9gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Reshape_1Reshape5gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Sum_17gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:??????????
?
@gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/tuple/group_depsNoOp8^gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Reshape:^gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Reshape_1
?
Hgradients/mlp/dropout_1/cond/dropout/Mul_1_grad/tuple/control_dependencyIdentity7gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/ReshapeA^gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Reshape*(
_output_shapes
:??????????
?
Jgradients/mlp/dropout_1/cond/dropout/Mul_1_grad/tuple/control_dependency_1Identity9gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Reshape_1A^gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/mlp/dropout_1/cond/dropout/Mul_1_grad/Reshape_1*(
_output_shapes
:??????????
?
gradients/Switch_2Switch	mlp/Elu_1mlp/dropout_1/cond/pred_id*
T0*<
_output_shapes*
(:??????????:??????????
i
gradients/Identity_2Identitygradients/Switch_2:1*
T0*(
_output_shapes
:??????????
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
_output_shapes
:*
out_type0
s
gradients/zeros_2/ConstConst^gradients/Identity_2*
_output_shapes
: *
dtype0*
valueB
 *    
?
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*(
_output_shapes
:??????????*

index_type0
?
;gradients/mlp/dropout_1/cond/Identity/Switch_grad/cond_gradMerge@gradients/mlp/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*
N*
T0**
_output_shapes
:??????????: 
?
3gradients/mlp/dropout_1/cond/dropout/Mul_grad/ShapeShape'mlp/dropout_1/cond/dropout/Mul/Switch:1*
T0*
_output_shapes
:*
out_type0
?
5gradients/mlp/dropout_1/cond/dropout/Mul_grad/Shape_1Shape mlp/dropout_1/cond/dropout/Const*
T0*
_output_shapes
: *
out_type0
?
Cgradients/mlp/dropout_1/cond/dropout/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/mlp/dropout_1/cond/dropout/Mul_grad/Shape5gradients/mlp/dropout_1/cond/dropout/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients/mlp/dropout_1/cond/dropout/Mul_grad/MulMulHgradients/mlp/dropout_1/cond/dropout/Mul_1_grad/tuple/control_dependency mlp/dropout_1/cond/dropout/Const*
T0*(
_output_shapes
:??????????
?
1gradients/mlp/dropout_1/cond/dropout/Mul_grad/SumSum1gradients/mlp/dropout_1/cond/dropout/Mul_grad/MulCgradients/mlp/dropout_1/cond/dropout/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
5gradients/mlp/dropout_1/cond/dropout/Mul_grad/ReshapeReshape1gradients/mlp/dropout_1/cond/dropout/Mul_grad/Sum3gradients/mlp/dropout_1/cond/dropout/Mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout_1/cond/dropout/Mul_grad/Mul_1Mul'mlp/dropout_1/cond/dropout/Mul/Switch:1Hgradients/mlp/dropout_1/cond/dropout/Mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout_1/cond/dropout/Mul_grad/Sum_1Sum3gradients/mlp/dropout_1/cond/dropout/Mul_grad/Mul_1Egradients/mlp/dropout_1/cond/dropout/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
7gradients/mlp/dropout_1/cond/dropout/Mul_grad/Reshape_1Reshape3gradients/mlp/dropout_1/cond/dropout/Mul_grad/Sum_15gradients/mlp/dropout_1/cond/dropout/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
>gradients/mlp/dropout_1/cond/dropout/Mul_grad/tuple/group_depsNoOp6^gradients/mlp/dropout_1/cond/dropout/Mul_grad/Reshape8^gradients/mlp/dropout_1/cond/dropout/Mul_grad/Reshape_1
?
Fgradients/mlp/dropout_1/cond/dropout/Mul_grad/tuple/control_dependencyIdentity5gradients/mlp/dropout_1/cond/dropout/Mul_grad/Reshape?^gradients/mlp/dropout_1/cond/dropout/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/mlp/dropout_1/cond/dropout/Mul_grad/Reshape*(
_output_shapes
:??????????
?
Hgradients/mlp/dropout_1/cond/dropout/Mul_grad/tuple/control_dependency_1Identity7gradients/mlp/dropout_1/cond/dropout/Mul_grad/Reshape_1?^gradients/mlp/dropout_1/cond/dropout/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mlp/dropout_1/cond/dropout/Mul_grad/Reshape_1*
_output_shapes
: 
?
gradients/Switch_3Switch	mlp/Elu_1mlp/dropout_1/cond/pred_id*
T0*<
_output_shapes*
(:??????????:??????????
g
gradients/Identity_3Identitygradients/Switch_3*
T0*(
_output_shapes
:??????????
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
_output_shapes
:*
out_type0
s
gradients/zeros_3/ConstConst^gradients/Identity_3*
_output_shapes
: *
dtype0*
valueB
 *    
?
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*(
_output_shapes
:??????????*

index_type0
?
>gradients/mlp/dropout_1/cond/dropout/Mul/Switch_grad/cond_gradMergegradients/zeros_3Fgradients/mlp/dropout_1/cond/dropout/Mul_grad/tuple/control_dependency*
N*
T0**
_output_shapes
:??????????: 
?
gradients/AddN_2AddN;gradients/mlp/dropout_1/cond/Identity/Switch_grad/cond_grad>gradients/mlp/dropout_1/cond/dropout/Mul/Switch_grad/cond_grad*
N*
T0*N
_classD
B@loc:@gradients/mlp/dropout_1/cond/Identity/Switch_grad/cond_grad*(
_output_shapes
:??????????
{
 gradients/mlp/Elu_1_grad/EluGradEluGradgradients/AddN_2	mlp/Elu_1*
T0*(
_output_shapes
:??????????
?
1gradients/mlp/mlp_layer2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/mlp/Elu_1_grad/EluGrad*
T0*
_output_shapes	
:?*
data_formatNHWC
?
6gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/group_depsNoOp!^gradients/mlp/Elu_1_grad/EluGrad2^gradients/mlp/mlp_layer2/BiasAdd_grad/BiasAddGrad
?
>gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/control_dependencyIdentity gradients/mlp/Elu_1_grad/EluGrad7^gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/mlp/Elu_1_grad/EluGrad*(
_output_shapes
:??????????
?
@gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/mlp/mlp_layer2/BiasAdd_grad/BiasAddGrad7^gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mlp/mlp_layer2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
+gradients/mlp/mlp_layer2/MatMul_grad/MatMulMatMul>gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/control_dependencymlp/mlp_layer2/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(
?
-gradients/mlp/mlp_layer2/MatMul_grad/MatMul_1MatMulmlp/dropout/cond/Merge>gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
??*
transpose_a(*
transpose_b( 
?
5gradients/mlp/mlp_layer2/MatMul_grad/tuple/group_depsNoOp,^gradients/mlp/mlp_layer2/MatMul_grad/MatMul.^gradients/mlp/mlp_layer2/MatMul_grad/MatMul_1
?
=gradients/mlp/mlp_layer2/MatMul_grad/tuple/control_dependencyIdentity+gradients/mlp/mlp_layer2/MatMul_grad/MatMul6^gradients/mlp/mlp_layer2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
?gradients/mlp/mlp_layer2/MatMul_grad/tuple/control_dependency_1Identity-gradients/mlp/mlp_layer2/MatMul_grad/MatMul_16^gradients/mlp/mlp_layer2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mlp/mlp_layer2/MatMul_grad/MatMul_1* 
_output_shapes
:
??
?
/gradients/mlp/dropout/cond/Merge_grad/cond_gradSwitch=gradients/mlp/mlp_layer2/MatMul_grad/tuple/control_dependencymlp/dropout/cond/pred_id*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer2/MatMul_grad/MatMul*<
_output_shapes*
(:??????????:??????????
p
6gradients/mlp/dropout/cond/Merge_grad/tuple/group_depsNoOp0^gradients/mlp/dropout/cond/Merge_grad/cond_grad
?
>gradients/mlp/dropout/cond/Merge_grad/tuple/control_dependencyIdentity/gradients/mlp/dropout/cond/Merge_grad/cond_grad7^gradients/mlp/dropout/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
@gradients/mlp/dropout/cond/Merge_grad/tuple/control_dependency_1Identity1gradients/mlp/dropout/cond/Merge_grad/cond_grad:17^gradients/mlp/dropout/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
3gradients/mlp/dropout/cond/dropout/Mul_1_grad/ShapeShapemlp/dropout/cond/dropout/Mul*
T0*
_output_shapes
:*
out_type0
?
5gradients/mlp/dropout/cond/dropout/Mul_1_grad/Shape_1Shapemlp/dropout/cond/dropout/Cast*
T0*
_output_shapes
:*
out_type0
?
Cgradients/mlp/dropout/cond/dropout/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/mlp/dropout/cond/dropout/Mul_1_grad/Shape5gradients/mlp/dropout/cond/dropout/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients/mlp/dropout/cond/dropout/Mul_1_grad/MulMul@gradients/mlp/dropout/cond/Merge_grad/tuple/control_dependency_1mlp/dropout/cond/dropout/Cast*
T0*(
_output_shapes
:??????????
?
1gradients/mlp/dropout/cond/dropout/Mul_1_grad/SumSum1gradients/mlp/dropout/cond/dropout/Mul_1_grad/MulCgradients/mlp/dropout/cond/dropout/Mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
5gradients/mlp/dropout/cond/dropout/Mul_1_grad/ReshapeReshape1gradients/mlp/dropout/cond/dropout/Mul_1_grad/Sum3gradients/mlp/dropout/cond/dropout/Mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout/cond/dropout/Mul_1_grad/Mul_1Mulmlp/dropout/cond/dropout/Mul@gradients/mlp/dropout/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:??????????
?
3gradients/mlp/dropout/cond/dropout/Mul_1_grad/Sum_1Sum3gradients/mlp/dropout/cond/dropout/Mul_1_grad/Mul_1Egradients/mlp/dropout/cond/dropout/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
7gradients/mlp/dropout/cond/dropout/Mul_1_grad/Reshape_1Reshape3gradients/mlp/dropout/cond/dropout/Mul_1_grad/Sum_15gradients/mlp/dropout/cond/dropout/Mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:??????????
?
>gradients/mlp/dropout/cond/dropout/Mul_1_grad/tuple/group_depsNoOp6^gradients/mlp/dropout/cond/dropout/Mul_1_grad/Reshape8^gradients/mlp/dropout/cond/dropout/Mul_1_grad/Reshape_1
?
Fgradients/mlp/dropout/cond/dropout/Mul_1_grad/tuple/control_dependencyIdentity5gradients/mlp/dropout/cond/dropout/Mul_1_grad/Reshape?^gradients/mlp/dropout/cond/dropout/Mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/mlp/dropout/cond/dropout/Mul_1_grad/Reshape*(
_output_shapes
:??????????
?
Hgradients/mlp/dropout/cond/dropout/Mul_1_grad/tuple/control_dependency_1Identity7gradients/mlp/dropout/cond/dropout/Mul_1_grad/Reshape_1?^gradients/mlp/dropout/cond/dropout/Mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mlp/dropout/cond/dropout/Mul_1_grad/Reshape_1*(
_output_shapes
:??????????
?
gradients/Switch_4Switchmlp/Elumlp/dropout/cond/pred_id*
T0*<
_output_shapes*
(:??????????:??????????
i
gradients/Identity_4Identitygradients/Switch_4:1*
T0*(
_output_shapes
:??????????
e
gradients/Shape_5Shapegradients/Switch_4:1*
T0*
_output_shapes
:*
out_type0
s
gradients/zeros_4/ConstConst^gradients/Identity_4*
_output_shapes
: *
dtype0*
valueB
 *    
?
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*(
_output_shapes
:??????????*

index_type0
?
9gradients/mlp/dropout/cond/Identity/Switch_grad/cond_gradMerge>gradients/mlp/dropout/cond/Merge_grad/tuple/control_dependencygradients/zeros_4*
N*
T0**
_output_shapes
:??????????: 
?
1gradients/mlp/dropout/cond/dropout/Mul_grad/ShapeShape%mlp/dropout/cond/dropout/Mul/Switch:1*
T0*
_output_shapes
:*
out_type0
?
3gradients/mlp/dropout/cond/dropout/Mul_grad/Shape_1Shapemlp/dropout/cond/dropout/Const*
T0*
_output_shapes
: *
out_type0
?
Agradients/mlp/dropout/cond/dropout/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/mlp/dropout/cond/dropout/Mul_grad/Shape3gradients/mlp/dropout/cond/dropout/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
/gradients/mlp/dropout/cond/dropout/Mul_grad/MulMulFgradients/mlp/dropout/cond/dropout/Mul_1_grad/tuple/control_dependencymlp/dropout/cond/dropout/Const*
T0*(
_output_shapes
:??????????
?
/gradients/mlp/dropout/cond/dropout/Mul_grad/SumSum/gradients/mlp/dropout/cond/dropout/Mul_grad/MulAgradients/mlp/dropout/cond/dropout/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
3gradients/mlp/dropout/cond/dropout/Mul_grad/ReshapeReshape/gradients/mlp/dropout/cond/dropout/Mul_grad/Sum1gradients/mlp/dropout/cond/dropout/Mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
1gradients/mlp/dropout/cond/dropout/Mul_grad/Mul_1Mul%mlp/dropout/cond/dropout/Mul/Switch:1Fgradients/mlp/dropout/cond/dropout/Mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
1gradients/mlp/dropout/cond/dropout/Mul_grad/Sum_1Sum1gradients/mlp/dropout/cond/dropout/Mul_grad/Mul_1Cgradients/mlp/dropout/cond/dropout/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
5gradients/mlp/dropout/cond/dropout/Mul_grad/Reshape_1Reshape1gradients/mlp/dropout/cond/dropout/Mul_grad/Sum_13gradients/mlp/dropout/cond/dropout/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
<gradients/mlp/dropout/cond/dropout/Mul_grad/tuple/group_depsNoOp4^gradients/mlp/dropout/cond/dropout/Mul_grad/Reshape6^gradients/mlp/dropout/cond/dropout/Mul_grad/Reshape_1
?
Dgradients/mlp/dropout/cond/dropout/Mul_grad/tuple/control_dependencyIdentity3gradients/mlp/dropout/cond/dropout/Mul_grad/Reshape=^gradients/mlp/dropout/cond/dropout/Mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/mlp/dropout/cond/dropout/Mul_grad/Reshape*(
_output_shapes
:??????????
?
Fgradients/mlp/dropout/cond/dropout/Mul_grad/tuple/control_dependency_1Identity5gradients/mlp/dropout/cond/dropout/Mul_grad/Reshape_1=^gradients/mlp/dropout/cond/dropout/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/mlp/dropout/cond/dropout/Mul_grad/Reshape_1*
_output_shapes
: 
?
gradients/Switch_5Switchmlp/Elumlp/dropout/cond/pred_id*
T0*<
_output_shapes*
(:??????????:??????????
g
gradients/Identity_5Identitygradients/Switch_5*
T0*(
_output_shapes
:??????????
c
gradients/Shape_6Shapegradients/Switch_5*
T0*
_output_shapes
:*
out_type0
s
gradients/zeros_5/ConstConst^gradients/Identity_5*
_output_shapes
: *
dtype0*
valueB
 *    
?
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*(
_output_shapes
:??????????*

index_type0
?
<gradients/mlp/dropout/cond/dropout/Mul/Switch_grad/cond_gradMergegradients/zeros_5Dgradients/mlp/dropout/cond/dropout/Mul_grad/tuple/control_dependency*
N*
T0**
_output_shapes
:??????????: 
?
gradients/AddN_3AddN9gradients/mlp/dropout/cond/Identity/Switch_grad/cond_grad<gradients/mlp/dropout/cond/dropout/Mul/Switch_grad/cond_grad*
N*
T0*L
_classB
@>loc:@gradients/mlp/dropout/cond/Identity/Switch_grad/cond_grad*(
_output_shapes
:??????????
w
gradients/mlp/Elu_grad/EluGradEluGradgradients/AddN_3mlp/Elu*
T0*(
_output_shapes
:??????????
?
1gradients/mlp/mlp_layer1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/mlp/Elu_grad/EluGrad*
T0*
_output_shapes	
:?*
data_formatNHWC
?
6gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/group_depsNoOp^gradients/mlp/Elu_grad/EluGrad2^gradients/mlp/mlp_layer1/BiasAdd_grad/BiasAddGrad
?
>gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/mlp/Elu_grad/EluGrad7^gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mlp/Elu_grad/EluGrad*(
_output_shapes
:??????????
?
@gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/mlp/mlp_layer1/BiasAdd_grad/BiasAddGrad7^gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mlp/mlp_layer1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
+gradients/mlp/mlp_layer1/MatMul_grad/MatMulMatMul>gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/control_dependencymlp/mlp_layer1/kernel/read*
T0*'
_output_shapes
:?????????P*
transpose_a( *
transpose_b(
?
-gradients/mlp/mlp_layer1/MatMul_grad/MatMul_1MatMulconcat_2>gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	P?*
transpose_a(*
transpose_b( 
?
5gradients/mlp/mlp_layer1/MatMul_grad/tuple/group_depsNoOp,^gradients/mlp/mlp_layer1/MatMul_grad/MatMul.^gradients/mlp/mlp_layer1/MatMul_grad/MatMul_1
?
=gradients/mlp/mlp_layer1/MatMul_grad/tuple/control_dependencyIdentity+gradients/mlp/mlp_layer1/MatMul_grad/MatMul6^gradients/mlp/mlp_layer1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/mlp/mlp_layer1/MatMul_grad/MatMul*'
_output_shapes
:?????????P
?
?gradients/mlp/mlp_layer1/MatMul_grad/tuple/control_dependency_1Identity-gradients/mlp/mlp_layer1/MatMul_grad/MatMul_16^gradients/mlp/mlp_layer1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mlp/mlp_layer1/MatMul_grad/MatMul_1*
_output_shapes
:	P?
^
gradients/concat_2_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
u
gradients/concat_2_grad/modFloorModconcat_2/axisgradients/concat_2_grad/Rank*
T0*
_output_shapes
: 
x
gradients/concat_2_grad/ShapeShapeembedding_lookup_4/Identity*
T0*
_output_shapes
:*
out_type0
?
gradients/concat_2_grad/ShapeNShapeNembedding_lookup_4/Identityembedding_lookup_5/IdentityReshape	Reshape_3*
N*
T0*,
_output_shapes
::::*
out_type0
?
$gradients/concat_2_grad/ConcatOffsetConcatOffsetgradients/concat_2_grad/modgradients/concat_2_grad/ShapeN gradients/concat_2_grad/ShapeN:1 gradients/concat_2_grad/ShapeN:2 gradients/concat_2_grad/ShapeN:3*
N*,
_output_shapes
::::
?
gradients/concat_2_grad/SliceSlice=gradients/mlp/mlp_layer1/MatMul_grad/tuple/control_dependency$gradients/concat_2_grad/ConcatOffsetgradients/concat_2_grad/ShapeN*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_2_grad/Slice_1Slice=gradients/mlp/mlp_layer1/MatMul_grad/tuple/control_dependency&gradients/concat_2_grad/ConcatOffset:1 gradients/concat_2_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_2_grad/Slice_2Slice=gradients/mlp/mlp_layer1/MatMul_grad/tuple/control_dependency&gradients/concat_2_grad/ConcatOffset:2 gradients/concat_2_grad/ShapeN:2*
Index0*
T0*'
_output_shapes
:?????????
?
gradients/concat_2_grad/Slice_3Slice=gradients/mlp/mlp_layer1/MatMul_grad/tuple/control_dependency&gradients/concat_2_grad/ConcatOffset:3 gradients/concat_2_grad/ShapeN:3*
Index0*
T0*'
_output_shapes
:????????? 
?
(gradients/concat_2_grad/tuple/group_depsNoOp^gradients/concat_2_grad/Slice ^gradients/concat_2_grad/Slice_1 ^gradients/concat_2_grad/Slice_2 ^gradients/concat_2_grad/Slice_3
?
0gradients/concat_2_grad/tuple/control_dependencyIdentitygradients/concat_2_grad/Slice)^gradients/concat_2_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_2_grad/Slice*'
_output_shapes
:?????????
?
2gradients/concat_2_grad/tuple/control_dependency_1Identitygradients/concat_2_grad/Slice_1)^gradients/concat_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_2_grad/Slice_1*'
_output_shapes
:?????????
?
2gradients/concat_2_grad/tuple/control_dependency_2Identitygradients/concat_2_grad/Slice_2)^gradients/concat_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_2_grad/Slice_2*'
_output_shapes
:?????????
?
2gradients/concat_2_grad/tuple/control_dependency_3Identitygradients/concat_2_grad/Slice_3)^gradients/concat_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_2_grad/Slice_3*'
_output_shapes
:????????? 
w
gradients/Reshape_grad/ShapeShapeembedding_lookup_7/Identity*
T0*
_output_shapes
:*
out_type0
?
gradients/Reshape_grad/ReshapeReshape2gradients/concat_2_grad/tuple/control_dependency_2gradients/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
c
gradients/Reshape_3_grad/ShapeShapeMul_1*
T0*
_output_shapes
:*
out_type0
?
 gradients/Reshape_3_grad/ReshapeReshape2gradients/concat_2_grad/tuple/control_dependency_3gradients/Reshape_3_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
'gradients/embedding_lookup_4_grad/ShapeConst*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0	*%
valueB	"?A             
?
&gradients/embedding_lookup_4_grad/CastCast'gradients/embedding_lookup_4_grad/Shape*

DstT0*

SrcT0	*
Truncate( *"
_class
loc:@embed_user_feat*
_output_shapes
:
n
&gradients/embedding_lookup_4_grad/SizeSizePlaceholder_1*
T0*
_output_shapes
: *
out_type0
r
0gradients/embedding_lookup_4_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
,gradients/embedding_lookup_4_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_4_grad/Size0gradients/embedding_lookup_4_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

5gradients/embedding_lookup_4_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
7gradients/embedding_lookup_4_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
7gradients/embedding_lookup_4_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
/gradients/embedding_lookup_4_grad/strided_sliceStridedSlice&gradients/embedding_lookup_4_grad/Cast5gradients/embedding_lookup_4_grad/strided_slice/stack7gradients/embedding_lookup_4_grad/strided_slice/stack_17gradients/embedding_lookup_4_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
o
-gradients/embedding_lookup_4_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
(gradients/embedding_lookup_4_grad/concatConcatV2,gradients/embedding_lookup_4_grad/ExpandDims/gradients/embedding_lookup_4_grad/strided_slice-gradients/embedding_lookup_4_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
)gradients/embedding_lookup_4_grad/ReshapeReshape0gradients/concat_2_grad/tuple/control_dependency(gradients/embedding_lookup_4_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
+gradients/embedding_lookup_4_grad/Reshape_1ReshapePlaceholder_1,gradients/embedding_lookup_4_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
'gradients/embedding_lookup_5_grad/ShapeConst*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0	*%
valueB	"hC             
?
&gradients/embedding_lookup_5_grad/CastCast'gradients/embedding_lookup_5_grad/Shape*

DstT0*

SrcT0	*
Truncate( *"
_class
loc:@embed_item_feat*
_output_shapes
:
n
&gradients/embedding_lookup_5_grad/SizeSizePlaceholder_2*
T0*
_output_shapes
: *
out_type0
r
0gradients/embedding_lookup_5_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
,gradients/embedding_lookup_5_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_5_grad/Size0gradients/embedding_lookup_5_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

5gradients/embedding_lookup_5_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
7gradients/embedding_lookup_5_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
7gradients/embedding_lookup_5_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
/gradients/embedding_lookup_5_grad/strided_sliceStridedSlice&gradients/embedding_lookup_5_grad/Cast5gradients/embedding_lookup_5_grad/strided_slice/stack7gradients/embedding_lookup_5_grad/strided_slice/stack_17gradients/embedding_lookup_5_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
o
-gradients/embedding_lookup_5_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
(gradients/embedding_lookup_5_grad/concatConcatV2,gradients/embedding_lookup_5_grad/ExpandDims/gradients/embedding_lookup_5_grad/strided_slice-gradients/embedding_lookup_5_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
)gradients/embedding_lookup_5_grad/ReshapeReshape2gradients/concat_2_grad/tuple/control_dependency_1(gradients/embedding_lookup_5_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
+gradients/embedding_lookup_5_grad/Reshape_1ReshapePlaceholder_2,gradients/embedding_lookup_5_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
gradients/AddN_4AddN2gradients/concat_1_grad/tuple/control_dependency_2gradients/Reshape_grad/Reshape*
N*
T0*2
_class(
&$loc:@gradients/concat_1_grad/Slice_2*+
_output_shapes
:?????????
?
gradients/AddN_5AddN2gradients/concat_1_grad/tuple/control_dependency_3 gradients/Reshape_3_grad/Reshape*
N*
T0*2
_class(
&$loc:@gradients/concat_1_grad/Slice_3*+
_output_shapes
:?????????
`
gradients/Mul_1_grad/ShapeShapeTile_1*
T0*
_output_shapes
:*
out_type0
e
gradients/Mul_1_grad/Shape_1Shape	Reshape_1*
T0*
_output_shapes
:*
out_type0
?
*gradients/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_1_grad/Shapegradients/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
r
gradients/Mul_1_grad/MulMulgradients/AddN_5	Reshape_1*
T0*+
_output_shapes
:?????????
?
gradients/Mul_1_grad/SumSumgradients/Mul_1_grad/Mul*gradients/Mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/Mul_1_grad/ReshapeReshapegradients/Mul_1_grad/Sumgradients/Mul_1_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
q
gradients/Mul_1_grad/Mul_1MulTile_1gradients/AddN_5*
T0*+
_output_shapes
:?????????
?
gradients/Mul_1_grad/Sum_1Sumgradients/Mul_1_grad/Mul_1,gradients/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/Mul_1_grad/Reshape_1Reshapegradients/Mul_1_grad/Sum_1gradients/Mul_1_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:?????????
m
%gradients/Mul_1_grad/tuple/group_depsNoOp^gradients/Mul_1_grad/Reshape^gradients/Mul_1_grad/Reshape_1
?
-gradients/Mul_1_grad/tuple/control_dependencyIdentitygradients/Mul_1_grad/Reshape&^gradients/Mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Mul_1_grad/Reshape*+
_output_shapes
:?????????
?
/gradients/Mul_1_grad/tuple/control_dependency_1Identitygradients/Mul_1_grad/Reshape_1&^gradients/Mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Mul_1_grad/Reshape_1*+
_output_shapes
:?????????
W
gradients/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gradients/concatConcatV2)gradients/embedding_lookup_2_grad/Reshape)gradients/embedding_lookup_4_grad/Reshapegradients/concat/axis*
N*
T0*

Tidx0*'
_output_shapes
:?????????
Y
gradients/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gradients/concat_1ConcatV2+gradients/embedding_lookup_2_grad/Reshape_1+gradients/embedding_lookup_4_grad/Reshape_1gradients/concat_1/axis*
N*
T0*

Tidx0*#
_output_shapes
:?????????
Y
gradients/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gradients/concat_2ConcatV2)gradients/embedding_lookup_3_grad/Reshape)gradients/embedding_lookup_5_grad/Reshapegradients/concat_2/axis*
N*
T0*

Tidx0*'
_output_shapes
:?????????
Y
gradients/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gradients/concat_3ConcatV2+gradients/embedding_lookup_3_grad/Reshape_1+gradients/embedding_lookup_5_grad/Reshape_1gradients/concat_3/axis*
N*
T0*

Tidx0*#
_output_shapes
:?????????
?
'gradients/embedding_lookup_7_grad/ShapeConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
:*
dtype0	*%
valueB	"=              
?
&gradients/embedding_lookup_7_grad/CastCast'gradients/embedding_lookup_7_grad/Shape*

DstT0*

SrcT0	*
Truncate( *$
_class
loc:@embed_sparse_feat*
_output_shapes
:
n
&gradients/embedding_lookup_7_grad/SizeSizePlaceholder_3*
T0*
_output_shapes
: *
out_type0
r
0gradients/embedding_lookup_7_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
,gradients/embedding_lookup_7_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_7_grad/Size0gradients/embedding_lookup_7_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

5gradients/embedding_lookup_7_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
7gradients/embedding_lookup_7_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
7gradients/embedding_lookup_7_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
/gradients/embedding_lookup_7_grad/strided_sliceStridedSlice&gradients/embedding_lookup_7_grad/Cast5gradients/embedding_lookup_7_grad/strided_slice/stack7gradients/embedding_lookup_7_grad/strided_slice/stack_17gradients/embedding_lookup_7_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
o
-gradients/embedding_lookup_7_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
(gradients/embedding_lookup_7_grad/concatConcatV2,gradients/embedding_lookup_7_grad/ExpandDims/gradients/embedding_lookup_7_grad/strided_slice-gradients/embedding_lookup_7_grad/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
)gradients/embedding_lookup_7_grad/ReshapeReshapegradients/AddN_4(gradients/embedding_lookup_7_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
+gradients/embedding_lookup_7_grad/Reshape_1ReshapePlaceholder_3,gradients/embedding_lookup_7_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
p
gradients/Tile_1_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         
?
gradients/Tile_1_grad/stackPackTile_1/multiplesgradients/Tile_1_grad/Shape*
N*
T0*
_output_shapes

:*

axis 
u
$gradients/Tile_1_grad/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
?
gradients/Tile_1_grad/transpose	Transposegradients/Tile_1_grad/stack$gradients/Tile_1_grad/transpose/perm*
T0*
Tperm0*
_output_shapes

:
v
#gradients/Tile_1_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
gradients/Tile_1_grad/ReshapeReshapegradients/Tile_1_grad/transpose#gradients/Tile_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
\
gradients/Tile_1_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
c
!gradients/Tile_1_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
c
!gradients/Tile_1_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Tile_1_grad/rangeRange!gradients/Tile_1_grad/range/startgradients/Tile_1_grad/Size!gradients/Tile_1_grad/range/delta*

Tidx0*
_output_shapes
:
?
gradients/Tile_1_grad/Reshape_1Reshape-gradients/Mul_1_grad/tuple/control_dependencygradients/Tile_1_grad/Reshape*
T0*
Tshape0*d
_output_shapesR
P:N??????????????????????????????????????????????????????
?
gradients/Tile_1_grad/SumSumgradients/Tile_1_grad/Reshape_1gradients/Tile_1_grad/range*
T0*

Tidx0*"
_output_shapes
:*
	keep_dims( 
r
!gradients/ExpandDims_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
?
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/Tile_1_grad/Sum!gradients/ExpandDims_2_grad/Shape*
T0*
Tshape0*
_output_shapes

:
}
beta1_power/initial_valueConst*
_class
loc:@dense/bias*
_output_shapes
: *
dtype0*
valueB
 *fff?
?
beta1_power
VariableV2*
_class
loc:@dense/bias*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
i
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
}
beta2_power/initial_valueConst*
_class
loc:@dense/bias*
_output_shapes
: *
dtype0*
valueB
 *w??
?
beta2_power
VariableV2*
_class
loc:@dense/bias*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class
loc:@dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
i
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
7linear_user_feat/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@linear_user_feat*
_output_shapes
:*
dtype0*
valueB"?A     
?
-linear_user_feat/Adam/Initializer/zeros/ConstConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
'linear_user_feat/Adam/Initializer/zerosFill7linear_user_feat/Adam/Initializer/zeros/shape_as_tensor-linear_user_feat/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*

index_type0
?
linear_user_feat/Adam
VariableV2*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
	container *
dtype0*
shape:
Ã*
shared_name 
?
linear_user_feat/Adam/AssignAssignlinear_user_feat/Adam'linear_user_feat/Adam/Initializer/zeros*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
linear_user_feat/Adam/readIdentitylinear_user_feat/Adam*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
9linear_user_feat/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@linear_user_feat*
_output_shapes
:*
dtype0*
valueB"?A     
?
/linear_user_feat/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
)linear_user_feat/Adam_1/Initializer/zerosFill9linear_user_feat/Adam_1/Initializer/zeros/shape_as_tensor/linear_user_feat/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*

index_type0
?
linear_user_feat/Adam_1
VariableV2*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
	container *
dtype0*
shape:
Ã*
shared_name 
?
linear_user_feat/Adam_1/AssignAssignlinear_user_feat/Adam_1)linear_user_feat/Adam_1/Initializer/zeros*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
linear_user_feat/Adam_1/readIdentitylinear_user_feat/Adam_1*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
7linear_item_feat/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@linear_item_feat*
_output_shapes
:*
dtype0*
valueB"hC     
?
-linear_item_feat/Adam/Initializer/zeros/ConstConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
'linear_item_feat/Adam/Initializer/zerosFill7linear_item_feat/Adam/Initializer/zeros/shape_as_tensor-linear_item_feat/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*

index_type0
?
linear_item_feat/Adam
VariableV2*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
linear_item_feat/Adam/AssignAssignlinear_item_feat/Adam'linear_item_feat/Adam/Initializer/zeros*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
linear_item_feat/Adam/readIdentitylinear_item_feat/Adam*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
9linear_item_feat/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@linear_item_feat*
_output_shapes
:*
dtype0*
valueB"hC     
?
/linear_item_feat/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
)linear_item_feat/Adam_1/Initializer/zerosFill9linear_item_feat/Adam_1/Initializer/zeros/shape_as_tensor/linear_item_feat/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*

index_type0
?
linear_item_feat/Adam_1
VariableV2*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
linear_item_feat/Adam_1/AssignAssignlinear_item_feat/Adam_1)linear_item_feat/Adam_1/Initializer/zeros*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
linear_item_feat/Adam_1/readIdentitylinear_item_feat/Adam_1*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
6embed_user_feat/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0*
valueB"?A     
?
,embed_user_feat/Adam/Initializer/zeros/ConstConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
&embed_user_feat/Adam/Initializer/zerosFill6embed_user_feat/Adam/Initializer/zeros/shape_as_tensor,embed_user_feat/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*

index_type0
?
embed_user_feat/Adam
VariableV2*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
	container *
dtype0*
shape:
Ã*
shared_name 
?
embed_user_feat/Adam/AssignAssignembed_user_feat/Adam&embed_user_feat/Adam/Initializer/zeros*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
embed_user_feat/Adam/readIdentityembed_user_feat/Adam*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
8embed_user_feat/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0*
valueB"?A     
?
.embed_user_feat/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
(embed_user_feat/Adam_1/Initializer/zerosFill8embed_user_feat/Adam_1/Initializer/zeros/shape_as_tensor.embed_user_feat/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*

index_type0
?
embed_user_feat/Adam_1
VariableV2*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
	container *
dtype0*
shape:
Ã*
shared_name 
?
embed_user_feat/Adam_1/AssignAssignembed_user_feat/Adam_1(embed_user_feat/Adam_1/Initializer/zeros*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
embed_user_feat/Adam_1/readIdentityembed_user_feat/Adam_1*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
6embed_item_feat/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0*
valueB"hC     
?
,embed_item_feat/Adam/Initializer/zeros/ConstConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
&embed_item_feat/Adam/Initializer/zerosFill6embed_item_feat/Adam/Initializer/zeros/shape_as_tensor,embed_item_feat/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*

index_type0
?
embed_item_feat/Adam
VariableV2*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
embed_item_feat/Adam/AssignAssignembed_item_feat/Adam&embed_item_feat/Adam/Initializer/zeros*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
embed_item_feat/Adam/readIdentityembed_item_feat/Adam*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
8embed_item_feat/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0*
valueB"hC     
?
.embed_item_feat/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *    
?
(embed_item_feat/Adam_1/Initializer/zerosFill8embed_item_feat/Adam_1/Initializer/zeros/shape_as_tensor.embed_item_feat/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*

index_type0
?
embed_item_feat/Adam_1
VariableV2*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
embed_item_feat/Adam_1/AssignAssignembed_item_feat/Adam_1(embed_item_feat/Adam_1/Initializer/zeros*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
embed_item_feat/Adam_1/readIdentityembed_item_feat/Adam_1*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
)linear_sparse_feat/Adam/Initializer/zerosConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
dtype0*
valueB=*    
?
linear_sparse_feat/Adam
VariableV2*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
	container *
dtype0*
shape:=*
shared_name 
?
linear_sparse_feat/Adam/AssignAssignlinear_sparse_feat/Adam)linear_sparse_feat/Adam/Initializer/zeros*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking(*
validate_shape(
?
linear_sparse_feat/Adam/readIdentitylinear_sparse_feat/Adam*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
+linear_sparse_feat/Adam_1/Initializer/zerosConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
dtype0*
valueB=*    
?
linear_sparse_feat/Adam_1
VariableV2*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
	container *
dtype0*
shape:=*
shared_name 
?
 linear_sparse_feat/Adam_1/AssignAssignlinear_sparse_feat/Adam_1+linear_sparse_feat/Adam_1/Initializer/zeros*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking(*
validate_shape(
?
linear_sparse_feat/Adam_1/readIdentitylinear_sparse_feat/Adam_1*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
(embed_sparse_feat/Adam/Initializer/zerosConst*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
dtype0*
valueB=*    
?
embed_sparse_feat/Adam
VariableV2*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
	container *
dtype0*
shape
:=*
shared_name 
?
embed_sparse_feat/Adam/AssignAssignembed_sparse_feat/Adam(embed_sparse_feat/Adam/Initializer/zeros*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking(*
validate_shape(
?
embed_sparse_feat/Adam/readIdentityembed_sparse_feat/Adam*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
*embed_sparse_feat/Adam_1/Initializer/zerosConst*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
dtype0*
valueB=*    
?
embed_sparse_feat/Adam_1
VariableV2*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
	container *
dtype0*
shape
:=*
shared_name 
?
embed_sparse_feat/Adam_1/AssignAssignembed_sparse_feat/Adam_1*embed_sparse_feat/Adam_1/Initializer/zeros*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking(*
validate_shape(
?
embed_sparse_feat/Adam_1/readIdentityembed_sparse_feat/Adam_1*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
(linear_dense_feat/Adam/Initializer/zerosConst*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
dtype0*
valueB*    
?
linear_dense_feat/Adam
VariableV2*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
linear_dense_feat/Adam/AssignAssignlinear_dense_feat/Adam(linear_dense_feat/Adam/Initializer/zeros*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
use_locking(*
validate_shape(
?
linear_dense_feat/Adam/readIdentitylinear_dense_feat/Adam*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:
?
*linear_dense_feat/Adam_1/Initializer/zerosConst*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
dtype0*
valueB*    
?
linear_dense_feat/Adam_1
VariableV2*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
linear_dense_feat/Adam_1/AssignAssignlinear_dense_feat/Adam_1*linear_dense_feat/Adam_1/Initializer/zeros*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
use_locking(*
validate_shape(
?
linear_dense_feat/Adam_1/readIdentitylinear_dense_feat/Adam_1*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:
?
'embed_dense_feat/Adam/Initializer/zerosConst*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
dtype0*
valueB*    
?
embed_dense_feat/Adam
VariableV2*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
	container *
dtype0*
shape
:*
shared_name 
?
embed_dense_feat/Adam/AssignAssignembed_dense_feat/Adam'embed_dense_feat/Adam/Initializer/zeros*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
use_locking(*
validate_shape(
?
embed_dense_feat/Adam/readIdentityembed_dense_feat/Adam*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:
?
)embed_dense_feat/Adam_1/Initializer/zerosConst*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
dtype0*
valueB*    
?
embed_dense_feat/Adam_1
VariableV2*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
	container *
dtype0*
shape
:*
shared_name 
?
embed_dense_feat/Adam_1/AssignAssignembed_dense_feat/Adam_1)embed_dense_feat/Adam_1/Initializer/zeros*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
use_locking(*
validate_shape(
?
embed_dense_feat/Adam_1/readIdentityembed_dense_feat/Adam_1*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:
?
#dense/kernel/Adam/Initializer/zerosConst*
_class
loc:@dense/kernel*
_output_shapes

:*
dtype0*
valueB*    
?
dense/kernel/Adam
VariableV2*
_class
loc:@dense/kernel*
_output_shapes

:*
	container *
dtype0*
shape
:*
shared_name 
?
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(

dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
%dense/kernel/Adam_1/Initializer/zerosConst*
_class
loc:@dense/kernel*
_output_shapes

:*
dtype0*
valueB*    
?
dense/kernel/Adam_1
VariableV2*
_class
loc:@dense/kernel*
_output_shapes

:*
	container *
dtype0*
shape
:*
shared_name 
?
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
?
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
!dense/bias/Adam/Initializer/zerosConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense/bias/Adam
VariableV2*
_class
loc:@dense/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
u
dense/bias/Adam/readIdentitydense/bias/Adam*
T0*
_class
loc:@dense/bias*
_output_shapes
:
?
#dense/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense/bias/Adam_1
VariableV2*
_class
loc:@dense/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
y
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias*
_output_shapes
:
?
<mlp/mlp_layer1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:*
dtype0*
valueB"P      
?
2mlp/mlp_layer1/kernel/Adam/Initializer/zeros/ConstConst*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
,mlp/mlp_layer1/kernel/Adam/Initializer/zerosFill<mlp/mlp_layer1/kernel/Adam/Initializer/zeros/shape_as_tensor2mlp/mlp_layer1/kernel/Adam/Initializer/zeros/Const*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*

index_type0
?
mlp/mlp_layer1/kernel/Adam
VariableV2*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
	container *
dtype0*
shape:	P?*
shared_name 
?
!mlp/mlp_layer1/kernel/Adam/AssignAssignmlp/mlp_layer1/kernel/Adam,mlp/mlp_layer1/kernel/Adam/Initializer/zeros*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
use_locking(*
validate_shape(
?
mlp/mlp_layer1/kernel/Adam/readIdentitymlp/mlp_layer1/kernel/Adam*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?
?
>mlp/mlp_layer1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:*
dtype0*
valueB"P      
?
4mlp/mlp_layer1/kernel/Adam_1/Initializer/zeros/ConstConst*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
.mlp/mlp_layer1/kernel/Adam_1/Initializer/zerosFill>mlp/mlp_layer1/kernel/Adam_1/Initializer/zeros/shape_as_tensor4mlp/mlp_layer1/kernel/Adam_1/Initializer/zeros/Const*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*

index_type0
?
mlp/mlp_layer1/kernel/Adam_1
VariableV2*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
	container *
dtype0*
shape:	P?*
shared_name 
?
#mlp/mlp_layer1/kernel/Adam_1/AssignAssignmlp/mlp_layer1/kernel/Adam_1.mlp/mlp_layer1/kernel/Adam_1/Initializer/zeros*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
use_locking(*
validate_shape(
?
!mlp/mlp_layer1/kernel/Adam_1/readIdentitymlp/mlp_layer1/kernel/Adam_1*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?
?
*mlp/mlp_layer1/bias/Adam/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer1/bias/Adam
VariableV2*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
mlp/mlp_layer1/bias/Adam/AssignAssignmlp/mlp_layer1/bias/Adam*mlp/mlp_layer1/bias/Adam/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer1/bias/Adam/readIdentitymlp/mlp_layer1/bias/Adam*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?
?
,mlp/mlp_layer1/bias/Adam_1/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer1/bias/Adam_1
VariableV2*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
!mlp/mlp_layer1/bias/Adam_1/AssignAssignmlp/mlp_layer1/bias/Adam_1,mlp/mlp_layer1/bias/Adam_1/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer1/bias/Adam_1/readIdentitymlp/mlp_layer1/bias/Adam_1*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?
?
<mlp/mlp_layer2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
2mlp/mlp_layer2/kernel/Adam/Initializer/zeros/ConstConst*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
,mlp/mlp_layer2/kernel/Adam/Initializer/zerosFill<mlp/mlp_layer2/kernel/Adam/Initializer/zeros/shape_as_tensor2mlp/mlp_layer2/kernel/Adam/Initializer/zeros/Const*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*

index_type0
?
mlp/mlp_layer2/kernel/Adam
VariableV2*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
!mlp/mlp_layer2/kernel/Adam/AssignAssignmlp/mlp_layer2/kernel/Adam,mlp/mlp_layer2/kernel/Adam/Initializer/zeros*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
mlp/mlp_layer2/kernel/Adam/readIdentitymlp/mlp_layer2/kernel/Adam*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??
?
>mlp/mlp_layer2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
4mlp/mlp_layer2/kernel/Adam_1/Initializer/zeros/ConstConst*(
_class
loc:@mlp/mlp_layer2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
.mlp/mlp_layer2/kernel/Adam_1/Initializer/zerosFill>mlp/mlp_layer2/kernel/Adam_1/Initializer/zeros/shape_as_tensor4mlp/mlp_layer2/kernel/Adam_1/Initializer/zeros/Const*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*

index_type0
?
mlp/mlp_layer2/kernel/Adam_1
VariableV2*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
#mlp/mlp_layer2/kernel/Adam_1/AssignAssignmlp/mlp_layer2/kernel/Adam_1.mlp/mlp_layer2/kernel/Adam_1/Initializer/zeros*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
!mlp/mlp_layer2/kernel/Adam_1/readIdentitymlp/mlp_layer2/kernel/Adam_1*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??
?
*mlp/mlp_layer2/bias/Adam/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer2/bias/Adam
VariableV2*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
mlp/mlp_layer2/bias/Adam/AssignAssignmlp/mlp_layer2/bias/Adam*mlp/mlp_layer2/bias/Adam/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer2/bias/Adam/readIdentitymlp/mlp_layer2/bias/Adam*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?
?
,mlp/mlp_layer2/bias/Adam_1/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer2/bias/Adam_1
VariableV2*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
!mlp/mlp_layer2/bias/Adam_1/AssignAssignmlp/mlp_layer2/bias/Adam_1,mlp/mlp_layer2/bias/Adam_1/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer2/bias/Adam_1/readIdentitymlp/mlp_layer2/bias/Adam_1*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?
?
<mlp/mlp_layer3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
2mlp/mlp_layer3/kernel/Adam/Initializer/zeros/ConstConst*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
,mlp/mlp_layer3/kernel/Adam/Initializer/zerosFill<mlp/mlp_layer3/kernel/Adam/Initializer/zeros/shape_as_tensor2mlp/mlp_layer3/kernel/Adam/Initializer/zeros/Const*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*

index_type0
?
mlp/mlp_layer3/kernel/Adam
VariableV2*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
!mlp/mlp_layer3/kernel/Adam/AssignAssignmlp/mlp_layer3/kernel/Adam,mlp/mlp_layer3/kernel/Adam/Initializer/zeros*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
mlp/mlp_layer3/kernel/Adam/readIdentitymlp/mlp_layer3/kernel/Adam*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??
?
>mlp/mlp_layer3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
4mlp/mlp_layer3/kernel/Adam_1/Initializer/zeros/ConstConst*(
_class
loc:@mlp/mlp_layer3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
.mlp/mlp_layer3/kernel/Adam_1/Initializer/zerosFill>mlp/mlp_layer3/kernel/Adam_1/Initializer/zeros/shape_as_tensor4mlp/mlp_layer3/kernel/Adam_1/Initializer/zeros/Const*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*

index_type0
?
mlp/mlp_layer3/kernel/Adam_1
VariableV2*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
	container *
dtype0*
shape:
??*
shared_name 
?
#mlp/mlp_layer3/kernel/Adam_1/AssignAssignmlp/mlp_layer3/kernel/Adam_1.mlp/mlp_layer3/kernel/Adam_1/Initializer/zeros*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
!mlp/mlp_layer3/kernel/Adam_1/readIdentitymlp/mlp_layer3/kernel/Adam_1*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??
?
*mlp/mlp_layer3/bias/Adam/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer3/bias/Adam
VariableV2*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
mlp/mlp_layer3/bias/Adam/AssignAssignmlp/mlp_layer3/bias/Adam*mlp/mlp_layer3/bias/Adam/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer3/bias/Adam/readIdentitymlp/mlp_layer3/bias/Adam*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?
?
,mlp/mlp_layer3/bias/Adam_1/Initializer/zerosConst*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
mlp/mlp_layer3/bias/Adam_1
VariableV2*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
	container *
dtype0*
shape:?*
shared_name 
?
!mlp/mlp_layer3/bias/Adam_1/AssignAssignmlp/mlp_layer3/bias/Adam_1,mlp/mlp_layer3/bias/Adam_1/Initializer/zeros*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
mlp/mlp_layer3/bias/Adam_1/readIdentitymlp/mlp_layer3/bias/Adam_1*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?
?
%dense_1/kernel/Adam/Initializer/zerosConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
dtype0*
valueB	?*    
?
dense_1/kernel/Adam
VariableV2*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
	container *
dtype0*
shape:	?*
shared_name 
?
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adam%dense_1/kernel/Adam/Initializer/zeros*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
?
'dense_1/kernel/Adam_1/Initializer/zerosConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
dtype0*
valueB	?*    
?
dense_1/kernel/Adam_1
VariableV2*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
	container *
dtype0*
shape:	?*
shared_name 
?
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1'dense_1/kernel/Adam_1/Initializer/zeros*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
?
#dense_1/bias/Adam/Initializer/zerosConst*
_class
loc:@dense_1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense_1/bias/Adam
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense_1/bias/Adam/AssignAssigndense_1/bias/Adam#dense_1/bias/Adam/Initializer/zeros*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
{
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
?
%dense_1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense_1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense_1/bias/Adam_1
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
?
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1%dense_1/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(

dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
W
Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w??
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2
?
#Adam/update_linear_user_feat/UniqueUnique)gradients/embedding_lookup_grad/Reshape_1*
T0*#
_class
loc:@linear_user_feat*2
_output_shapes 
:?????????:?????????*
out_idx0
?
"Adam/update_linear_user_feat/ShapeShape#Adam/update_linear_user_feat/Unique*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
:*
out_type0
?
0Adam/update_linear_user_feat/strided_slice/stackConst*#
_class
loc:@linear_user_feat*
_output_shapes
:*
dtype0*
valueB: 
?
2Adam/update_linear_user_feat/strided_slice/stack_1Const*#
_class
loc:@linear_user_feat*
_output_shapes
:*
dtype0*
valueB:
?
2Adam/update_linear_user_feat/strided_slice/stack_2Const*#
_class
loc:@linear_user_feat*
_output_shapes
:*
dtype0*
valueB:
?
*Adam/update_linear_user_feat/strided_sliceStridedSlice"Adam/update_linear_user_feat/Shape0Adam/update_linear_user_feat/strided_slice/stack2Adam/update_linear_user_feat/strided_slice/stack_12Adam/update_linear_user_feat/strided_slice/stack_2*
Index0*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
/Adam/update_linear_user_feat/UnsortedSegmentSumUnsortedSegmentSum'gradients/embedding_lookup_grad/Reshape%Adam/update_linear_user_feat/Unique:1*Adam/update_linear_user_feat/strided_slice*
T0*
Tindices0*
Tnumsegments0*#
_class
loc:@linear_user_feat*'
_output_shapes
:?????????
?
"Adam/update_linear_user_feat/sub/xConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
 Adam/update_linear_user_feat/subSub"Adam/update_linear_user_feat/sub/xbeta2_power/read*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: 
?
!Adam/update_linear_user_feat/SqrtSqrt Adam/update_linear_user_feat/sub*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: 
?
 Adam/update_linear_user_feat/mulMulAdam/learning_rate!Adam/update_linear_user_feat/Sqrt*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: 
?
$Adam/update_linear_user_feat/sub_1/xConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"Adam/update_linear_user_feat/sub_1Sub$Adam/update_linear_user_feat/sub_1/xbeta1_power/read*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: 
?
$Adam/update_linear_user_feat/truedivRealDiv Adam/update_linear_user_feat/mul"Adam/update_linear_user_feat/sub_1*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: 
?
$Adam/update_linear_user_feat/sub_2/xConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"Adam/update_linear_user_feat/sub_2Sub$Adam/update_linear_user_feat/sub_2/x
Adam/beta1*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: 
?
"Adam/update_linear_user_feat/mul_1Mul/Adam/update_linear_user_feat/UnsortedSegmentSum"Adam/update_linear_user_feat/sub_2*
T0*#
_class
loc:@linear_user_feat*'
_output_shapes
:?????????
?
"Adam/update_linear_user_feat/mul_2Mullinear_user_feat/Adam/read
Adam/beta1*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
#Adam/update_linear_user_feat/AssignAssignlinear_user_feat/Adam"Adam/update_linear_user_feat/mul_2*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking( *
validate_shape(
?
'Adam/update_linear_user_feat/ScatterAdd
ScatterAddlinear_user_feat/Adam#Adam/update_linear_user_feat/Unique"Adam/update_linear_user_feat/mul_1$^Adam/update_linear_user_feat/Assign*
T0*
Tindices0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking( 
?
"Adam/update_linear_user_feat/mul_3Mul/Adam/update_linear_user_feat/UnsortedSegmentSum/Adam/update_linear_user_feat/UnsortedSegmentSum*
T0*#
_class
loc:@linear_user_feat*'
_output_shapes
:?????????
?
$Adam/update_linear_user_feat/sub_3/xConst*#
_class
loc:@linear_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"Adam/update_linear_user_feat/sub_3Sub$Adam/update_linear_user_feat/sub_3/x
Adam/beta2*
T0*#
_class
loc:@linear_user_feat*
_output_shapes
: 
?
"Adam/update_linear_user_feat/mul_4Mul"Adam/update_linear_user_feat/mul_3"Adam/update_linear_user_feat/sub_3*
T0*#
_class
loc:@linear_user_feat*'
_output_shapes
:?????????
?
"Adam/update_linear_user_feat/mul_5Mullinear_user_feat/Adam_1/read
Adam/beta2*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
%Adam/update_linear_user_feat/Assign_1Assignlinear_user_feat/Adam_1"Adam/update_linear_user_feat/mul_5*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking( *
validate_shape(
?
)Adam/update_linear_user_feat/ScatterAdd_1
ScatterAddlinear_user_feat/Adam_1#Adam/update_linear_user_feat/Unique"Adam/update_linear_user_feat/mul_4&^Adam/update_linear_user_feat/Assign_1*
T0*
Tindices0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking( 
?
#Adam/update_linear_user_feat/Sqrt_1Sqrt)Adam/update_linear_user_feat/ScatterAdd_1*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
"Adam/update_linear_user_feat/mul_6Mul$Adam/update_linear_user_feat/truediv'Adam/update_linear_user_feat/ScatterAdd*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
 Adam/update_linear_user_feat/addAddV2#Adam/update_linear_user_feat/Sqrt_1Adam/epsilon*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
&Adam/update_linear_user_feat/truediv_1RealDiv"Adam/update_linear_user_feat/mul_6 Adam/update_linear_user_feat/add*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã
?
&Adam/update_linear_user_feat/AssignSub	AssignSublinear_user_feat&Adam/update_linear_user_feat/truediv_1*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking( 
?
'Adam/update_linear_user_feat/group_depsNoOp'^Adam/update_linear_user_feat/AssignSub(^Adam/update_linear_user_feat/ScatterAdd*^Adam/update_linear_user_feat/ScatterAdd_1*#
_class
loc:@linear_user_feat
?
#Adam/update_linear_item_feat/UniqueUnique+gradients/embedding_lookup_1_grad/Reshape_1*
T0*#
_class
loc:@linear_item_feat*2
_output_shapes 
:?????????:?????????*
out_idx0
?
"Adam/update_linear_item_feat/ShapeShape#Adam/update_linear_item_feat/Unique*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
:*
out_type0
?
0Adam/update_linear_item_feat/strided_slice/stackConst*#
_class
loc:@linear_item_feat*
_output_shapes
:*
dtype0*
valueB: 
?
2Adam/update_linear_item_feat/strided_slice/stack_1Const*#
_class
loc:@linear_item_feat*
_output_shapes
:*
dtype0*
valueB:
?
2Adam/update_linear_item_feat/strided_slice/stack_2Const*#
_class
loc:@linear_item_feat*
_output_shapes
:*
dtype0*
valueB:
?
*Adam/update_linear_item_feat/strided_sliceStridedSlice"Adam/update_linear_item_feat/Shape0Adam/update_linear_item_feat/strided_slice/stack2Adam/update_linear_item_feat/strided_slice/stack_12Adam/update_linear_item_feat/strided_slice/stack_2*
Index0*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
/Adam/update_linear_item_feat/UnsortedSegmentSumUnsortedSegmentSum)gradients/embedding_lookup_1_grad/Reshape%Adam/update_linear_item_feat/Unique:1*Adam/update_linear_item_feat/strided_slice*
T0*
Tindices0*
Tnumsegments0*#
_class
loc:@linear_item_feat*'
_output_shapes
:?????????
?
"Adam/update_linear_item_feat/sub/xConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
 Adam/update_linear_item_feat/subSub"Adam/update_linear_item_feat/sub/xbeta2_power/read*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: 
?
!Adam/update_linear_item_feat/SqrtSqrt Adam/update_linear_item_feat/sub*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: 
?
 Adam/update_linear_item_feat/mulMulAdam/learning_rate!Adam/update_linear_item_feat/Sqrt*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: 
?
$Adam/update_linear_item_feat/sub_1/xConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"Adam/update_linear_item_feat/sub_1Sub$Adam/update_linear_item_feat/sub_1/xbeta1_power/read*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: 
?
$Adam/update_linear_item_feat/truedivRealDiv Adam/update_linear_item_feat/mul"Adam/update_linear_item_feat/sub_1*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: 
?
$Adam/update_linear_item_feat/sub_2/xConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"Adam/update_linear_item_feat/sub_2Sub$Adam/update_linear_item_feat/sub_2/x
Adam/beta1*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: 
?
"Adam/update_linear_item_feat/mul_1Mul/Adam/update_linear_item_feat/UnsortedSegmentSum"Adam/update_linear_item_feat/sub_2*
T0*#
_class
loc:@linear_item_feat*'
_output_shapes
:?????????
?
"Adam/update_linear_item_feat/mul_2Mullinear_item_feat/Adam/read
Adam/beta1*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
#Adam/update_linear_item_feat/AssignAssignlinear_item_feat/Adam"Adam/update_linear_item_feat/mul_2*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking( *
validate_shape(
?
'Adam/update_linear_item_feat/ScatterAdd
ScatterAddlinear_item_feat/Adam#Adam/update_linear_item_feat/Unique"Adam/update_linear_item_feat/mul_1$^Adam/update_linear_item_feat/Assign*
T0*
Tindices0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking( 
?
"Adam/update_linear_item_feat/mul_3Mul/Adam/update_linear_item_feat/UnsortedSegmentSum/Adam/update_linear_item_feat/UnsortedSegmentSum*
T0*#
_class
loc:@linear_item_feat*'
_output_shapes
:?????????
?
$Adam/update_linear_item_feat/sub_3/xConst*#
_class
loc:@linear_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"Adam/update_linear_item_feat/sub_3Sub$Adam/update_linear_item_feat/sub_3/x
Adam/beta2*
T0*#
_class
loc:@linear_item_feat*
_output_shapes
: 
?
"Adam/update_linear_item_feat/mul_4Mul"Adam/update_linear_item_feat/mul_3"Adam/update_linear_item_feat/sub_3*
T0*#
_class
loc:@linear_item_feat*'
_output_shapes
:?????????
?
"Adam/update_linear_item_feat/mul_5Mullinear_item_feat/Adam_1/read
Adam/beta2*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
%Adam/update_linear_item_feat/Assign_1Assignlinear_item_feat/Adam_1"Adam/update_linear_item_feat/mul_5*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking( *
validate_shape(
?
)Adam/update_linear_item_feat/ScatterAdd_1
ScatterAddlinear_item_feat/Adam_1#Adam/update_linear_item_feat/Unique"Adam/update_linear_item_feat/mul_4&^Adam/update_linear_item_feat/Assign_1*
T0*
Tindices0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking( 
?
#Adam/update_linear_item_feat/Sqrt_1Sqrt)Adam/update_linear_item_feat/ScatterAdd_1*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
"Adam/update_linear_item_feat/mul_6Mul$Adam/update_linear_item_feat/truediv'Adam/update_linear_item_feat/ScatterAdd*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
 Adam/update_linear_item_feat/addAddV2#Adam/update_linear_item_feat/Sqrt_1Adam/epsilon*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
&Adam/update_linear_item_feat/truediv_1RealDiv"Adam/update_linear_item_feat/mul_6 Adam/update_linear_item_feat/add*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??
?
&Adam/update_linear_item_feat/AssignSub	AssignSublinear_item_feat&Adam/update_linear_item_feat/truediv_1*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking( 
?
'Adam/update_linear_item_feat/group_depsNoOp'^Adam/update_linear_item_feat/AssignSub(^Adam/update_linear_item_feat/ScatterAdd*^Adam/update_linear_item_feat/ScatterAdd_1*#
_class
loc:@linear_item_feat
?
"Adam/update_embed_user_feat/UniqueUniquegradients/concat_1*
T0*"
_class
loc:@embed_user_feat*2
_output_shapes 
:?????????:?????????*
out_idx0
?
!Adam/update_embed_user_feat/ShapeShape"Adam/update_embed_user_feat/Unique*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
:*
out_type0
?
/Adam/update_embed_user_feat/strided_slice/stackConst*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0*
valueB: 
?
1Adam/update_embed_user_feat/strided_slice/stack_1Const*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0*
valueB:
?
1Adam/update_embed_user_feat/strided_slice/stack_2Const*"
_class
loc:@embed_user_feat*
_output_shapes
:*
dtype0*
valueB:
?
)Adam/update_embed_user_feat/strided_sliceStridedSlice!Adam/update_embed_user_feat/Shape/Adam/update_embed_user_feat/strided_slice/stack1Adam/update_embed_user_feat/strided_slice/stack_11Adam/update_embed_user_feat/strided_slice/stack_2*
Index0*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
.Adam/update_embed_user_feat/UnsortedSegmentSumUnsortedSegmentSumgradients/concat$Adam/update_embed_user_feat/Unique:1)Adam/update_embed_user_feat/strided_slice*
T0*
Tindices0*
Tnumsegments0*"
_class
loc:@embed_user_feat*'
_output_shapes
:?????????
?
!Adam/update_embed_user_feat/sub/xConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Adam/update_embed_user_feat/subSub!Adam/update_embed_user_feat/sub/xbeta2_power/read*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: 
?
 Adam/update_embed_user_feat/SqrtSqrtAdam/update_embed_user_feat/sub*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: 
?
Adam/update_embed_user_feat/mulMulAdam/learning_rate Adam/update_embed_user_feat/Sqrt*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: 
?
#Adam/update_embed_user_feat/sub_1/xConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!Adam/update_embed_user_feat/sub_1Sub#Adam/update_embed_user_feat/sub_1/xbeta1_power/read*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: 
?
#Adam/update_embed_user_feat/truedivRealDivAdam/update_embed_user_feat/mul!Adam/update_embed_user_feat/sub_1*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: 
?
#Adam/update_embed_user_feat/sub_2/xConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!Adam/update_embed_user_feat/sub_2Sub#Adam/update_embed_user_feat/sub_2/x
Adam/beta1*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: 
?
!Adam/update_embed_user_feat/mul_1Mul.Adam/update_embed_user_feat/UnsortedSegmentSum!Adam/update_embed_user_feat/sub_2*
T0*"
_class
loc:@embed_user_feat*'
_output_shapes
:?????????
?
!Adam/update_embed_user_feat/mul_2Mulembed_user_feat/Adam/read
Adam/beta1*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
"Adam/update_embed_user_feat/AssignAssignembed_user_feat/Adam!Adam/update_embed_user_feat/mul_2*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking( *
validate_shape(
?
&Adam/update_embed_user_feat/ScatterAdd
ScatterAddembed_user_feat/Adam"Adam/update_embed_user_feat/Unique!Adam/update_embed_user_feat/mul_1#^Adam/update_embed_user_feat/Assign*
T0*
Tindices0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking( 
?
!Adam/update_embed_user_feat/mul_3Mul.Adam/update_embed_user_feat/UnsortedSegmentSum.Adam/update_embed_user_feat/UnsortedSegmentSum*
T0*"
_class
loc:@embed_user_feat*'
_output_shapes
:?????????
?
#Adam/update_embed_user_feat/sub_3/xConst*"
_class
loc:@embed_user_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!Adam/update_embed_user_feat/sub_3Sub#Adam/update_embed_user_feat/sub_3/x
Adam/beta2*
T0*"
_class
loc:@embed_user_feat*
_output_shapes
: 
?
!Adam/update_embed_user_feat/mul_4Mul!Adam/update_embed_user_feat/mul_3!Adam/update_embed_user_feat/sub_3*
T0*"
_class
loc:@embed_user_feat*'
_output_shapes
:?????????
?
!Adam/update_embed_user_feat/mul_5Mulembed_user_feat/Adam_1/read
Adam/beta2*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
$Adam/update_embed_user_feat/Assign_1Assignembed_user_feat/Adam_1!Adam/update_embed_user_feat/mul_5*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking( *
validate_shape(
?
(Adam/update_embed_user_feat/ScatterAdd_1
ScatterAddembed_user_feat/Adam_1"Adam/update_embed_user_feat/Unique!Adam/update_embed_user_feat/mul_4%^Adam/update_embed_user_feat/Assign_1*
T0*
Tindices0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking( 
?
"Adam/update_embed_user_feat/Sqrt_1Sqrt(Adam/update_embed_user_feat/ScatterAdd_1*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
!Adam/update_embed_user_feat/mul_6Mul#Adam/update_embed_user_feat/truediv&Adam/update_embed_user_feat/ScatterAdd*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
Adam/update_embed_user_feat/addAddV2"Adam/update_embed_user_feat/Sqrt_1Adam/epsilon*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
%Adam/update_embed_user_feat/truediv_1RealDiv!Adam/update_embed_user_feat/mul_6Adam/update_embed_user_feat/add*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã
?
%Adam/update_embed_user_feat/AssignSub	AssignSubembed_user_feat%Adam/update_embed_user_feat/truediv_1*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking( 
?
&Adam/update_embed_user_feat/group_depsNoOp&^Adam/update_embed_user_feat/AssignSub'^Adam/update_embed_user_feat/ScatterAdd)^Adam/update_embed_user_feat/ScatterAdd_1*"
_class
loc:@embed_user_feat
?
"Adam/update_embed_item_feat/UniqueUniquegradients/concat_3*
T0*"
_class
loc:@embed_item_feat*2
_output_shapes 
:?????????:?????????*
out_idx0
?
!Adam/update_embed_item_feat/ShapeShape"Adam/update_embed_item_feat/Unique*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
:*
out_type0
?
/Adam/update_embed_item_feat/strided_slice/stackConst*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0*
valueB: 
?
1Adam/update_embed_item_feat/strided_slice/stack_1Const*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0*
valueB:
?
1Adam/update_embed_item_feat/strided_slice/stack_2Const*"
_class
loc:@embed_item_feat*
_output_shapes
:*
dtype0*
valueB:
?
)Adam/update_embed_item_feat/strided_sliceStridedSlice!Adam/update_embed_item_feat/Shape/Adam/update_embed_item_feat/strided_slice/stack1Adam/update_embed_item_feat/strided_slice/stack_11Adam/update_embed_item_feat/strided_slice/stack_2*
Index0*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
.Adam/update_embed_item_feat/UnsortedSegmentSumUnsortedSegmentSumgradients/concat_2$Adam/update_embed_item_feat/Unique:1)Adam/update_embed_item_feat/strided_slice*
T0*
Tindices0*
Tnumsegments0*"
_class
loc:@embed_item_feat*'
_output_shapes
:?????????
?
!Adam/update_embed_item_feat/sub/xConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Adam/update_embed_item_feat/subSub!Adam/update_embed_item_feat/sub/xbeta2_power/read*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: 
?
 Adam/update_embed_item_feat/SqrtSqrtAdam/update_embed_item_feat/sub*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: 
?
Adam/update_embed_item_feat/mulMulAdam/learning_rate Adam/update_embed_item_feat/Sqrt*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: 
?
#Adam/update_embed_item_feat/sub_1/xConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!Adam/update_embed_item_feat/sub_1Sub#Adam/update_embed_item_feat/sub_1/xbeta1_power/read*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: 
?
#Adam/update_embed_item_feat/truedivRealDivAdam/update_embed_item_feat/mul!Adam/update_embed_item_feat/sub_1*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: 
?
#Adam/update_embed_item_feat/sub_2/xConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!Adam/update_embed_item_feat/sub_2Sub#Adam/update_embed_item_feat/sub_2/x
Adam/beta1*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: 
?
!Adam/update_embed_item_feat/mul_1Mul.Adam/update_embed_item_feat/UnsortedSegmentSum!Adam/update_embed_item_feat/sub_2*
T0*"
_class
loc:@embed_item_feat*'
_output_shapes
:?????????
?
!Adam/update_embed_item_feat/mul_2Mulembed_item_feat/Adam/read
Adam/beta1*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
"Adam/update_embed_item_feat/AssignAssignembed_item_feat/Adam!Adam/update_embed_item_feat/mul_2*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking( *
validate_shape(
?
&Adam/update_embed_item_feat/ScatterAdd
ScatterAddembed_item_feat/Adam"Adam/update_embed_item_feat/Unique!Adam/update_embed_item_feat/mul_1#^Adam/update_embed_item_feat/Assign*
T0*
Tindices0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking( 
?
!Adam/update_embed_item_feat/mul_3Mul.Adam/update_embed_item_feat/UnsortedSegmentSum.Adam/update_embed_item_feat/UnsortedSegmentSum*
T0*"
_class
loc:@embed_item_feat*'
_output_shapes
:?????????
?
#Adam/update_embed_item_feat/sub_3/xConst*"
_class
loc:@embed_item_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!Adam/update_embed_item_feat/sub_3Sub#Adam/update_embed_item_feat/sub_3/x
Adam/beta2*
T0*"
_class
loc:@embed_item_feat*
_output_shapes
: 
?
!Adam/update_embed_item_feat/mul_4Mul!Adam/update_embed_item_feat/mul_3!Adam/update_embed_item_feat/sub_3*
T0*"
_class
loc:@embed_item_feat*'
_output_shapes
:?????????
?
!Adam/update_embed_item_feat/mul_5Mulembed_item_feat/Adam_1/read
Adam/beta2*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
$Adam/update_embed_item_feat/Assign_1Assignembed_item_feat/Adam_1!Adam/update_embed_item_feat/mul_5*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking( *
validate_shape(
?
(Adam/update_embed_item_feat/ScatterAdd_1
ScatterAddembed_item_feat/Adam_1"Adam/update_embed_item_feat/Unique!Adam/update_embed_item_feat/mul_4%^Adam/update_embed_item_feat/Assign_1*
T0*
Tindices0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking( 
?
"Adam/update_embed_item_feat/Sqrt_1Sqrt(Adam/update_embed_item_feat/ScatterAdd_1*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
!Adam/update_embed_item_feat/mul_6Mul#Adam/update_embed_item_feat/truediv&Adam/update_embed_item_feat/ScatterAdd*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
Adam/update_embed_item_feat/addAddV2"Adam/update_embed_item_feat/Sqrt_1Adam/epsilon*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
%Adam/update_embed_item_feat/truediv_1RealDiv!Adam/update_embed_item_feat/mul_6Adam/update_embed_item_feat/add*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??
?
%Adam/update_embed_item_feat/AssignSub	AssignSubembed_item_feat%Adam/update_embed_item_feat/truediv_1*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking( 
?
&Adam/update_embed_item_feat/group_depsNoOp&^Adam/update_embed_item_feat/AssignSub'^Adam/update_embed_item_feat/ScatterAdd)^Adam/update_embed_item_feat/ScatterAdd_1*"
_class
loc:@embed_item_feat
?
%Adam/update_linear_sparse_feat/UniqueUnique+gradients/embedding_lookup_6_grad/Reshape_1*
T0*%
_class
loc:@linear_sparse_feat*2
_output_shapes 
:?????????:?????????*
out_idx0
?
$Adam/update_linear_sparse_feat/ShapeShape%Adam/update_linear_sparse_feat/Unique*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:*
out_type0
?
2Adam/update_linear_sparse_feat/strided_slice/stackConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
:*
dtype0*
valueB: 
?
4Adam/update_linear_sparse_feat/strided_slice/stack_1Const*%
_class
loc:@linear_sparse_feat*
_output_shapes
:*
dtype0*
valueB:
?
4Adam/update_linear_sparse_feat/strided_slice/stack_2Const*%
_class
loc:@linear_sparse_feat*
_output_shapes
:*
dtype0*
valueB:
?
,Adam/update_linear_sparse_feat/strided_sliceStridedSlice$Adam/update_linear_sparse_feat/Shape2Adam/update_linear_sparse_feat/strided_slice/stack4Adam/update_linear_sparse_feat/strided_slice/stack_14Adam/update_linear_sparse_feat/strided_slice/stack_2*
Index0*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
1Adam/update_linear_sparse_feat/UnsortedSegmentSumUnsortedSegmentSum)gradients/embedding_lookup_6_grad/Reshape'Adam/update_linear_sparse_feat/Unique:1,Adam/update_linear_sparse_feat/strided_slice*
T0*
Tindices0*
Tnumsegments0*%
_class
loc:@linear_sparse_feat*#
_output_shapes
:?????????
?
$Adam/update_linear_sparse_feat/sub/xConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"Adam/update_linear_sparse_feat/subSub$Adam/update_linear_sparse_feat/sub/xbeta2_power/read*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: 
?
#Adam/update_linear_sparse_feat/SqrtSqrt"Adam/update_linear_sparse_feat/sub*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: 
?
"Adam/update_linear_sparse_feat/mulMulAdam/learning_rate#Adam/update_linear_sparse_feat/Sqrt*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: 
?
&Adam/update_linear_sparse_feat/sub_1/xConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
$Adam/update_linear_sparse_feat/sub_1Sub&Adam/update_linear_sparse_feat/sub_1/xbeta1_power/read*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: 
?
&Adam/update_linear_sparse_feat/truedivRealDiv"Adam/update_linear_sparse_feat/mul$Adam/update_linear_sparse_feat/sub_1*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: 
?
&Adam/update_linear_sparse_feat/sub_2/xConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
$Adam/update_linear_sparse_feat/sub_2Sub&Adam/update_linear_sparse_feat/sub_2/x
Adam/beta1*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: 
?
$Adam/update_linear_sparse_feat/mul_1Mul1Adam/update_linear_sparse_feat/UnsortedSegmentSum$Adam/update_linear_sparse_feat/sub_2*
T0*%
_class
loc:@linear_sparse_feat*#
_output_shapes
:?????????
?
$Adam/update_linear_sparse_feat/mul_2Mullinear_sparse_feat/Adam/read
Adam/beta1*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
%Adam/update_linear_sparse_feat/AssignAssignlinear_sparse_feat/Adam$Adam/update_linear_sparse_feat/mul_2*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking( *
validate_shape(
?
)Adam/update_linear_sparse_feat/ScatterAdd
ScatterAddlinear_sparse_feat/Adam%Adam/update_linear_sparse_feat/Unique$Adam/update_linear_sparse_feat/mul_1&^Adam/update_linear_sparse_feat/Assign*
T0*
Tindices0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking( 
?
$Adam/update_linear_sparse_feat/mul_3Mul1Adam/update_linear_sparse_feat/UnsortedSegmentSum1Adam/update_linear_sparse_feat/UnsortedSegmentSum*
T0*%
_class
loc:@linear_sparse_feat*#
_output_shapes
:?????????
?
&Adam/update_linear_sparse_feat/sub_3/xConst*%
_class
loc:@linear_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
$Adam/update_linear_sparse_feat/sub_3Sub&Adam/update_linear_sparse_feat/sub_3/x
Adam/beta2*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
: 
?
$Adam/update_linear_sparse_feat/mul_4Mul$Adam/update_linear_sparse_feat/mul_3$Adam/update_linear_sparse_feat/sub_3*
T0*%
_class
loc:@linear_sparse_feat*#
_output_shapes
:?????????
?
$Adam/update_linear_sparse_feat/mul_5Mullinear_sparse_feat/Adam_1/read
Adam/beta2*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
'Adam/update_linear_sparse_feat/Assign_1Assignlinear_sparse_feat/Adam_1$Adam/update_linear_sparse_feat/mul_5*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking( *
validate_shape(
?
+Adam/update_linear_sparse_feat/ScatterAdd_1
ScatterAddlinear_sparse_feat/Adam_1%Adam/update_linear_sparse_feat/Unique$Adam/update_linear_sparse_feat/mul_4(^Adam/update_linear_sparse_feat/Assign_1*
T0*
Tindices0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking( 
?
%Adam/update_linear_sparse_feat/Sqrt_1Sqrt+Adam/update_linear_sparse_feat/ScatterAdd_1*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
$Adam/update_linear_sparse_feat/mul_6Mul&Adam/update_linear_sparse_feat/truediv)Adam/update_linear_sparse_feat/ScatterAdd*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
"Adam/update_linear_sparse_feat/addAddV2%Adam/update_linear_sparse_feat/Sqrt_1Adam/epsilon*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
(Adam/update_linear_sparse_feat/truediv_1RealDiv$Adam/update_linear_sparse_feat/mul_6"Adam/update_linear_sparse_feat/add*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=
?
(Adam/update_linear_sparse_feat/AssignSub	AssignSublinear_sparse_feat(Adam/update_linear_sparse_feat/truediv_1*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking( 
?
)Adam/update_linear_sparse_feat/group_depsNoOp)^Adam/update_linear_sparse_feat/AssignSub*^Adam/update_linear_sparse_feat/ScatterAdd,^Adam/update_linear_sparse_feat/ScatterAdd_1*%
_class
loc:@linear_sparse_feat
?
$Adam/update_embed_sparse_feat/UniqueUnique+gradients/embedding_lookup_7_grad/Reshape_1*
T0*$
_class
loc:@embed_sparse_feat*2
_output_shapes 
:?????????:?????????*
out_idx0
?
#Adam/update_embed_sparse_feat/ShapeShape$Adam/update_embed_sparse_feat/Unique*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
:*
out_type0
?
1Adam/update_embed_sparse_feat/strided_slice/stackConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
:*
dtype0*
valueB: 
?
3Adam/update_embed_sparse_feat/strided_slice/stack_1Const*$
_class
loc:@embed_sparse_feat*
_output_shapes
:*
dtype0*
valueB:
?
3Adam/update_embed_sparse_feat/strided_slice/stack_2Const*$
_class
loc:@embed_sparse_feat*
_output_shapes
:*
dtype0*
valueB:
?
+Adam/update_embed_sparse_feat/strided_sliceStridedSlice#Adam/update_embed_sparse_feat/Shape1Adam/update_embed_sparse_feat/strided_slice/stack3Adam/update_embed_sparse_feat/strided_slice/stack_13Adam/update_embed_sparse_feat/strided_slice/stack_2*
Index0*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
0Adam/update_embed_sparse_feat/UnsortedSegmentSumUnsortedSegmentSum)gradients/embedding_lookup_7_grad/Reshape&Adam/update_embed_sparse_feat/Unique:1+Adam/update_embed_sparse_feat/strided_slice*
T0*
Tindices0*
Tnumsegments0*$
_class
loc:@embed_sparse_feat*'
_output_shapes
:?????????
?
#Adam/update_embed_sparse_feat/sub/xConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!Adam/update_embed_sparse_feat/subSub#Adam/update_embed_sparse_feat/sub/xbeta2_power/read*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: 
?
"Adam/update_embed_sparse_feat/SqrtSqrt!Adam/update_embed_sparse_feat/sub*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: 
?
!Adam/update_embed_sparse_feat/mulMulAdam/learning_rate"Adam/update_embed_sparse_feat/Sqrt*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: 
?
%Adam/update_embed_sparse_feat/sub_1/xConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
#Adam/update_embed_sparse_feat/sub_1Sub%Adam/update_embed_sparse_feat/sub_1/xbeta1_power/read*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: 
?
%Adam/update_embed_sparse_feat/truedivRealDiv!Adam/update_embed_sparse_feat/mul#Adam/update_embed_sparse_feat/sub_1*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: 
?
%Adam/update_embed_sparse_feat/sub_2/xConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
#Adam/update_embed_sparse_feat/sub_2Sub%Adam/update_embed_sparse_feat/sub_2/x
Adam/beta1*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: 
?
#Adam/update_embed_sparse_feat/mul_1Mul0Adam/update_embed_sparse_feat/UnsortedSegmentSum#Adam/update_embed_sparse_feat/sub_2*
T0*$
_class
loc:@embed_sparse_feat*'
_output_shapes
:?????????
?
#Adam/update_embed_sparse_feat/mul_2Mulembed_sparse_feat/Adam/read
Adam/beta1*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
$Adam/update_embed_sparse_feat/AssignAssignembed_sparse_feat/Adam#Adam/update_embed_sparse_feat/mul_2*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking( *
validate_shape(
?
(Adam/update_embed_sparse_feat/ScatterAdd
ScatterAddembed_sparse_feat/Adam$Adam/update_embed_sparse_feat/Unique#Adam/update_embed_sparse_feat/mul_1%^Adam/update_embed_sparse_feat/Assign*
T0*
Tindices0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking( 
?
#Adam/update_embed_sparse_feat/mul_3Mul0Adam/update_embed_sparse_feat/UnsortedSegmentSum0Adam/update_embed_sparse_feat/UnsortedSegmentSum*
T0*$
_class
loc:@embed_sparse_feat*'
_output_shapes
:?????????
?
%Adam/update_embed_sparse_feat/sub_3/xConst*$
_class
loc:@embed_sparse_feat*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
#Adam/update_embed_sparse_feat/sub_3Sub%Adam/update_embed_sparse_feat/sub_3/x
Adam/beta2*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes
: 
?
#Adam/update_embed_sparse_feat/mul_4Mul#Adam/update_embed_sparse_feat/mul_3#Adam/update_embed_sparse_feat/sub_3*
T0*$
_class
loc:@embed_sparse_feat*'
_output_shapes
:?????????
?
#Adam/update_embed_sparse_feat/mul_5Mulembed_sparse_feat/Adam_1/read
Adam/beta2*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
&Adam/update_embed_sparse_feat/Assign_1Assignembed_sparse_feat/Adam_1#Adam/update_embed_sparse_feat/mul_5*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking( *
validate_shape(
?
*Adam/update_embed_sparse_feat/ScatterAdd_1
ScatterAddembed_sparse_feat/Adam_1$Adam/update_embed_sparse_feat/Unique#Adam/update_embed_sparse_feat/mul_4'^Adam/update_embed_sparse_feat/Assign_1*
T0*
Tindices0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking( 
?
$Adam/update_embed_sparse_feat/Sqrt_1Sqrt*Adam/update_embed_sparse_feat/ScatterAdd_1*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
#Adam/update_embed_sparse_feat/mul_6Mul%Adam/update_embed_sparse_feat/truediv(Adam/update_embed_sparse_feat/ScatterAdd*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
!Adam/update_embed_sparse_feat/addAddV2$Adam/update_embed_sparse_feat/Sqrt_1Adam/epsilon*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
'Adam/update_embed_sparse_feat/truediv_1RealDiv#Adam/update_embed_sparse_feat/mul_6!Adam/update_embed_sparse_feat/add*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=
?
'Adam/update_embed_sparse_feat/AssignSub	AssignSubembed_sparse_feat'Adam/update_embed_sparse_feat/truediv_1*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking( 
?
(Adam/update_embed_sparse_feat/group_depsNoOp(^Adam/update_embed_sparse_feat/AssignSub)^Adam/update_embed_sparse_feat/ScatterAdd+^Adam/update_embed_sparse_feat/ScatterAdd_1*$
_class
loc:@embed_sparse_feat
?
'Adam/update_linear_dense_feat/ApplyAdam	ApplyAdamlinear_dense_featlinear_dense_feat/Adamlinear_dense_feat/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/Tile_grad/Sum*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
use_locking( *
use_nesterov( 
?
&Adam/update_embed_dense_feat/ApplyAdam	ApplyAdamembed_dense_featembed_dense_feat/Adamembed_dense_feat/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon#gradients/ExpandDims_2_grad/Reshape*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
use_locking( *
use_nesterov( 
?
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking( *
use_nesterov( 
?
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking( *
use_nesterov( 
?
+Adam/update_mlp/mlp_layer1/kernel/ApplyAdam	ApplyAdammlp/mlp_layer1/kernelmlp/mlp_layer1/kernel/Adammlp/mlp_layer1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/mlp/mlp_layer1/MatMul_grad/tuple/control_dependency_1*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
use_locking( *
use_nesterov( 
?
)Adam/update_mlp/mlp_layer1/bias/ApplyAdam	ApplyAdammlp/mlp_layer1/biasmlp/mlp_layer1/bias/Adammlp/mlp_layer1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/mlp/mlp_layer1/BiasAdd_grad/tuple/control_dependency_1*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
use_locking( *
use_nesterov( 
?
+Adam/update_mlp/mlp_layer2/kernel/ApplyAdam	ApplyAdammlp/mlp_layer2/kernelmlp/mlp_layer2/kernel/Adammlp/mlp_layer2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/mlp/mlp_layer2/MatMul_grad/tuple/control_dependency_1*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
use_locking( *
use_nesterov( 
?
)Adam/update_mlp/mlp_layer2/bias/ApplyAdam	ApplyAdammlp/mlp_layer2/biasmlp/mlp_layer2/bias/Adammlp/mlp_layer2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/mlp/mlp_layer2/BiasAdd_grad/tuple/control_dependency_1*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
use_locking( *
use_nesterov( 
?
+Adam/update_mlp/mlp_layer3/kernel/ApplyAdam	ApplyAdammlp/mlp_layer3/kernelmlp/mlp_layer3/kernel/Adammlp/mlp_layer3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/mlp/mlp_layer3/MatMul_grad/tuple/control_dependency_1*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
use_locking( *
use_nesterov( 
?
)Adam/update_mlp/mlp_layer3/bias/ApplyAdam	ApplyAdammlp/mlp_layer3/biasmlp/mlp_layer3/bias/Adammlp/mlp_layer3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/mlp/mlp_layer3/BiasAdd_grad/tuple/control_dependency_1*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
use_locking( *
use_nesterov( 
?
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
use_locking( *
use_nesterov( 
?
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking( *
use_nesterov( 
?
Adam/mulMulbeta1_power/read
Adam/beta1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam'^Adam/update_embed_dense_feat/ApplyAdam'^Adam/update_embed_item_feat/group_deps)^Adam/update_embed_sparse_feat/group_deps'^Adam/update_embed_user_feat/group_deps(^Adam/update_linear_dense_feat/ApplyAdam(^Adam/update_linear_item_feat/group_deps*^Adam/update_linear_sparse_feat/group_deps(^Adam/update_linear_user_feat/group_deps*^Adam/update_mlp/mlp_layer1/bias/ApplyAdam,^Adam/update_mlp/mlp_layer1/kernel/ApplyAdam*^Adam/update_mlp/mlp_layer2/bias/ApplyAdam,^Adam/update_mlp/mlp_layer2/kernel/ApplyAdam*^Adam/update_mlp/mlp_layer3/bias/ApplyAdam,^Adam/update_mlp/mlp_layer3/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
loc:@dense/bias*
_output_shapes
: *
use_locking( *
validate_shape(
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam'^Adam/update_embed_dense_feat/ApplyAdam'^Adam/update_embed_item_feat/group_deps)^Adam/update_embed_sparse_feat/group_deps'^Adam/update_embed_user_feat/group_deps(^Adam/update_linear_dense_feat/ApplyAdam(^Adam/update_linear_item_feat/group_deps*^Adam/update_linear_sparse_feat/group_deps(^Adam/update_linear_user_feat/group_deps*^Adam/update_mlp/mlp_layer1/bias/ApplyAdam,^Adam/update_mlp/mlp_layer1/kernel/ApplyAdam*^Adam/update_mlp/mlp_layer2/bias/ApplyAdam,^Adam/update_mlp/mlp_layer2/kernel/ApplyAdam*^Adam/update_mlp/mlp_layer3/bias/ApplyAdam,^Adam/update_mlp/mlp_layer3/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@dense/bias*
_output_shapes
: *
use_locking( *
validate_shape(
?
AdamNoOp^Adam/Assign^Adam/Assign_1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam'^Adam/update_embed_dense_feat/ApplyAdam'^Adam/update_embed_item_feat/group_deps)^Adam/update_embed_sparse_feat/group_deps'^Adam/update_embed_user_feat/group_deps(^Adam/update_linear_dense_feat/ApplyAdam(^Adam/update_linear_item_feat/group_deps*^Adam/update_linear_sparse_feat/group_deps(^Adam/update_linear_user_feat/group_deps*^Adam/update_mlp/mlp_layer1/bias/ApplyAdam,^Adam/update_mlp/mlp_layer1/kernel/ApplyAdam*^Adam/update_mlp/mlp_layer2/bias/ApplyAdam,^Adam/update_mlp/mlp_layer2/kernel/ApplyAdam*^Adam/update_mlp/mlp_layer3/bias/ApplyAdam,^Adam/update_mlp/mlp_layer3/kernel/ApplyAdam


group_depsNoOp^Adam
?
initNoOp^beta1_power/Assign^beta2_power/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^dense_1/bias/Adam/Assign^dense_1/bias/Adam_1/Assign^dense_1/bias/Assign^dense_1/kernel/Adam/Assign^dense_1/kernel/Adam_1/Assign^dense_1/kernel/Assign^embed_dense_feat/Adam/Assign^embed_dense_feat/Adam_1/Assign^embed_dense_feat/Assign^embed_item_feat/Adam/Assign^embed_item_feat/Adam_1/Assign^embed_item_feat/Assign^embed_sparse_feat/Adam/Assign ^embed_sparse_feat/Adam_1/Assign^embed_sparse_feat/Assign^embed_user_feat/Adam/Assign^embed_user_feat/Adam_1/Assign^embed_user_feat/Assign^linear_dense_feat/Adam/Assign ^linear_dense_feat/Adam_1/Assign^linear_dense_feat/Assign^linear_item_feat/Adam/Assign^linear_item_feat/Adam_1/Assign^linear_item_feat/Assign^linear_sparse_feat/Adam/Assign!^linear_sparse_feat/Adam_1/Assign^linear_sparse_feat/Assign^linear_user_feat/Adam/Assign^linear_user_feat/Adam_1/Assign^linear_user_feat/Assign ^mlp/mlp_layer1/bias/Adam/Assign"^mlp/mlp_layer1/bias/Adam_1/Assign^mlp/mlp_layer1/bias/Assign"^mlp/mlp_layer1/kernel/Adam/Assign$^mlp/mlp_layer1/kernel/Adam_1/Assign^mlp/mlp_layer1/kernel/Assign ^mlp/mlp_layer2/bias/Adam/Assign"^mlp/mlp_layer2/bias/Adam_1/Assign^mlp/mlp_layer2/bias/Assign"^mlp/mlp_layer2/kernel/Adam/Assign$^mlp/mlp_layer2/kernel/Adam_1/Assign^mlp/mlp_layer2/kernel/Assign ^mlp/mlp_layer3/bias/Adam/Assign"^mlp/mlp_layer3/bias/Adam_1/Assign^mlp/mlp_layer3/bias/Assign"^mlp/mlp_layer3/kernel/Adam/Assign$^mlp/mlp_layer3/kernel/Adam_1/Assign^mlp/mlp_layer3/kernel/Assign
M
range/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range/limitConst*
_output_shapes
: *
dtype0*
valueB	 :
M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
_
rangeRangerange/startrange/limitrange/delta*

Tidx0*
_output_shapes

:
O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
GatherV2GatherV2linear_user_feat/readrangeGatherV2/axis*
Taxis0*
Tindices0*
Tparams0* 
_output_shapes
:
*

batch_dims 
X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
t
MeanMeanGatherV2Mean/reduction_indices*
T0*

Tidx0*
_output_shapes

:*
	keep_dims(
a
ScatterUpdate/indicesConst*
_output_shapes
:*
dtype0*
valueB:
?
ScatterUpdateScatterUpdatelinear_user_featScatterUpdate/indicesMean*
T0*
Tindices0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking( 
O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
Q
range_1/limitConst*
_output_shapes
: *
dtype0*
valueB	 :??
O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
g
range_1Rangerange_1/startrange_1/limitrange_1/delta*

Tidx0*
_output_shapes

:??
Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?

GatherV2_1GatherV2linear_item_feat/readrange_1GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0* 
_output_shapes
:
??*

batch_dims 
Z
Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
z
Mean_1Mean
GatherV2_1Mean_1/reduction_indices*
T0*

Tidx0*
_output_shapes

:*
	keep_dims(
c
ScatterUpdate_1/indicesConst*
_output_shapes
:*
dtype0*
valueB:??
?
ScatterUpdate_1ScatterUpdatelinear_item_featScatterUpdate_1/indicesMean_1*
T0*
Tindices0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking( 
O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 
Q
range_2/limitConst*
_output_shapes
: *
dtype0*
valueB	 :
O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
g
range_2Rangerange_2/startrange_2/limitrange_2/delta*

Tidx0*
_output_shapes

:
Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?

GatherV2_2GatherV2embed_user_feat/readrange_2GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0* 
_output_shapes
:
*

batch_dims 
Z
Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
z
Mean_2Mean
GatherV2_2Mean_2/reduction_indices*
T0*

Tidx0*
_output_shapes

:*
	keep_dims(
c
ScatterUpdate_2/indicesConst*
_output_shapes
:*
dtype0*
valueB:
?
ScatterUpdate_2ScatterUpdateembed_user_featScatterUpdate_2/indicesMean_2*
T0*
Tindices0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking( 
O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : 
Q
range_3/limitConst*
_output_shapes
: *
dtype0*
valueB	 :??
O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :
g
range_3Rangerange_3/startrange_3/limitrange_3/delta*

Tidx0*
_output_shapes

:??
Q
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?

GatherV2_3GatherV2embed_item_feat/readrange_3GatherV2_3/axis*
Taxis0*
Tindices0*
Tparams0* 
_output_shapes
:
??*

batch_dims 
Z
Mean_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
z
Mean_3Mean
GatherV2_3Mean_3/reduction_indices*
T0*

Tidx0*
_output_shapes

:*
	keep_dims(
c
ScatterUpdate_3/indicesConst*
_output_shapes
:*
dtype0*
valueB:??
?
ScatterUpdate_3ScatterUpdateembed_item_featScatterUpdate_3/indicesMean_3*
T0*
Tindices0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking( 
O
range_4/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_4/limitConst*
_output_shapes
: *
dtype0*
value	B :<
O
range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e
range_4Rangerange_4/startrange_4/limitrange_4/delta*

Tidx0*
_output_shapes
:<
Q
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?

GatherV2_4GatherV2linear_sparse_feat/readrange_4GatherV2_4/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:<*

batch_dims 
Z
Mean_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
v
Mean_4Mean
GatherV2_4Mean_4/reduction_indices*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
i
ScatterNdUpdate/indicesConst*
_output_shapes

:*
dtype0*
valueB:<
?
ScatterNdUpdateScatterNdUpdatelinear_sparse_featScatterNdUpdate/indicesMean_4*
T0*
Tindices0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking(
O
range_5/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_5/limitConst*
_output_shapes
: *
dtype0*
value	B :<
O
range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e
range_5Rangerange_5/startrange_5/limitrange_5/delta*

Tidx0*
_output_shapes
:<
Q
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?

GatherV2_5GatherV2embed_sparse_feat/readrange_5GatherV2_5/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:<*

batch_dims 
Z
Mean_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
z
Mean_5Mean
GatherV2_5Mean_5/reduction_indices*
T0*

Tidx0*
_output_shapes

:*
	keep_dims(
k
ScatterNdUpdate_1/indicesConst*
_output_shapes

:*
dtype0*
valueB:<
?
ScatterNdUpdate_1ScatterNdUpdateembed_sparse_featScatterNdUpdate_1/indicesMean_5*
T0*
Tindices0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking(
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp\part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
w
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?

save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?	
value?	B?	8Bbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bembed_dense_featBembed_dense_feat/AdamBembed_dense_feat/Adam_1Bembed_item_featBembed_item_feat/AdamBembed_item_feat/Adam_1Bembed_sparse_featBembed_sparse_feat/AdamBembed_sparse_feat/Adam_1Bembed_user_featBembed_user_feat/AdamBembed_user_feat/Adam_1Blinear_dense_featBlinear_dense_feat/AdamBlinear_dense_feat/Adam_1Blinear_item_featBlinear_item_feat/AdamBlinear_item_feat/Adam_1Blinear_sparse_featBlinear_sparse_feat/AdamBlinear_sparse_feat/Adam_1Blinear_user_featBlinear_user_feat/AdamBlinear_user_feat/Adam_1Bmlp/mlp_layer1/biasBmlp/mlp_layer1/bias/AdamBmlp/mlp_layer1/bias/Adam_1Bmlp/mlp_layer1/kernelBmlp/mlp_layer1/kernel/AdamBmlp/mlp_layer1/kernel/Adam_1Bmlp/mlp_layer2/biasBmlp/mlp_layer2/bias/AdamBmlp/mlp_layer2/bias/Adam_1Bmlp/mlp_layer2/kernelBmlp/mlp_layer2/kernel/AdamBmlp/mlp_layer2/kernel/Adam_1Bmlp/mlp_layer3/biasBmlp/mlp_layer3/bias/AdamBmlp/mlp_layer3/bias/Adam_1Bmlp/mlp_layer3/kernelBmlp/mlp_layer3/kernel/AdamBmlp/mlp_layer3/kernel/Adam_1
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1embed_dense_featembed_dense_feat/Adamembed_dense_feat/Adam_1embed_item_featembed_item_feat/Adamembed_item_feat/Adam_1embed_sparse_featembed_sparse_feat/Adamembed_sparse_feat/Adam_1embed_user_featembed_user_feat/Adamembed_user_feat/Adam_1linear_dense_featlinear_dense_feat/Adamlinear_dense_feat/Adam_1linear_item_featlinear_item_feat/Adamlinear_item_feat/Adam_1linear_sparse_featlinear_sparse_feat/Adamlinear_sparse_feat/Adam_1linear_user_featlinear_user_feat/Adamlinear_user_feat/Adam_1mlp/mlp_layer1/biasmlp/mlp_layer1/bias/Adammlp/mlp_layer1/bias/Adam_1mlp/mlp_layer1/kernelmlp/mlp_layer1/kernel/Adammlp/mlp_layer1/kernel/Adam_1mlp/mlp_layer2/biasmlp/mlp_layer2/bias/Adammlp/mlp_layer2/bias/Adam_1mlp/mlp_layer2/kernelmlp/mlp_layer2/kernel/Adammlp/mlp_layer2/kernel/Adam_1mlp/mlp_layer3/biasmlp/mlp_layer3/bias/Adammlp/mlp_layer3/bias/Adam_1mlp/mlp_layer3/kernelmlp/mlp_layer3/kernel/Adammlp/mlp_layer3/kernel/Adam_1"/device:CPU:0*F
dtypes<
:28
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:*

axis 
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?

save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?	
value?	B?	8Bbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bembed_dense_featBembed_dense_feat/AdamBembed_dense_feat/Adam_1Bembed_item_featBembed_item_feat/AdamBembed_item_feat/Adam_1Bembed_sparse_featBembed_sparse_feat/AdamBembed_sparse_feat/Adam_1Bembed_user_featBembed_user_feat/AdamBembed_user_feat/Adam_1Blinear_dense_featBlinear_dense_feat/AdamBlinear_dense_feat/Adam_1Blinear_item_featBlinear_item_feat/AdamBlinear_item_feat/Adam_1Blinear_sparse_featBlinear_sparse_feat/AdamBlinear_sparse_feat/Adam_1Blinear_user_featBlinear_user_feat/AdamBlinear_user_feat/Adam_1Bmlp/mlp_layer1/biasBmlp/mlp_layer1/bias/AdamBmlp/mlp_layer1/bias/Adam_1Bmlp/mlp_layer1/kernelBmlp/mlp_layer1/kernel/AdamBmlp/mlp_layer1/kernel/Adam_1Bmlp/mlp_layer2/biasBmlp/mlp_layer2/bias/AdamBmlp/mlp_layer2/bias/Adam_1Bmlp/mlp_layer2/kernelBmlp/mlp_layer2/kernel/AdamBmlp/mlp_layer2/kernel/Adam_1Bmlp/mlp_layer3/biasBmlp/mlp_layer3/bias/AdamBmlp/mlp_layer3/bias/Adam_1Bmlp/mlp_layer3/kernelBmlp/mlp_layer3/kernel/AdamBmlp/mlp_layer3/kernel/Adam_1
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28
?
save/AssignAssignbeta1_powersave/RestoreV2*
T0*
_class
loc:@dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
?
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
T0*
_class
loc:@dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
?
save/Assign_2Assign
dense/biassave/RestoreV2:2*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_3Assigndense/bias/Adamsave/RestoreV2:3*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_4Assigndense/bias/Adam_1save/RestoreV2:4*
T0*
_class
loc:@dense/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_5Assigndense/kernelsave/RestoreV2:5*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
?
save/Assign_6Assigndense/kernel/Adamsave/RestoreV2:6*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
?
save/Assign_7Assigndense/kernel/Adam_1save/RestoreV2:7*
T0*
_class
loc:@dense/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
?
save/Assign_8Assigndense_1/biassave/RestoreV2:8*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_9Assigndense_1/bias/Adamsave/RestoreV2:9*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_10Assigndense_1/bias/Adam_1save/RestoreV2:10*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_11Assigndense_1/kernelsave/RestoreV2:11*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
save/Assign_12Assigndense_1/kernel/Adamsave/RestoreV2:12*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
save/Assign_13Assigndense_1/kernel/Adam_1save/RestoreV2:13*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
save/Assign_14Assignembed_dense_featsave/RestoreV2:14*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
use_locking(*
validate_shape(
?
save/Assign_15Assignembed_dense_feat/Adamsave/RestoreV2:15*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
use_locking(*
validate_shape(
?
save/Assign_16Assignembed_dense_feat/Adam_1save/RestoreV2:16*
T0*#
_class
loc:@embed_dense_feat*
_output_shapes

:*
use_locking(*
validate_shape(
?
save/Assign_17Assignembed_item_featsave/RestoreV2:17*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_18Assignembed_item_feat/Adamsave/RestoreV2:18*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_19Assignembed_item_feat/Adam_1save/RestoreV2:19*
T0*"
_class
loc:@embed_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_20Assignembed_sparse_featsave/RestoreV2:20*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking(*
validate_shape(
?
save/Assign_21Assignembed_sparse_feat/Adamsave/RestoreV2:21*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking(*
validate_shape(
?
save/Assign_22Assignembed_sparse_feat/Adam_1save/RestoreV2:22*
T0*$
_class
loc:@embed_sparse_feat*
_output_shapes

:=*
use_locking(*
validate_shape(
?
save/Assign_23Assignembed_user_featsave/RestoreV2:23*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
save/Assign_24Assignembed_user_feat/Adamsave/RestoreV2:24*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
save/Assign_25Assignembed_user_feat/Adam_1save/RestoreV2:25*
T0*"
_class
loc:@embed_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
save/Assign_26Assignlinear_dense_featsave/RestoreV2:26*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_27Assignlinear_dense_feat/Adamsave/RestoreV2:27*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_28Assignlinear_dense_feat/Adam_1save/RestoreV2:28*
T0*$
_class
loc:@linear_dense_feat*
_output_shapes
:*
use_locking(*
validate_shape(
?
save/Assign_29Assignlinear_item_featsave/RestoreV2:29*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_30Assignlinear_item_feat/Adamsave/RestoreV2:30*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_31Assignlinear_item_feat/Adam_1save/RestoreV2:31*
T0*#
_class
loc:@linear_item_feat* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_32Assignlinear_sparse_featsave/RestoreV2:32*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking(*
validate_shape(
?
save/Assign_33Assignlinear_sparse_feat/Adamsave/RestoreV2:33*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking(*
validate_shape(
?
save/Assign_34Assignlinear_sparse_feat/Adam_1save/RestoreV2:34*
T0*%
_class
loc:@linear_sparse_feat*
_output_shapes
:=*
use_locking(*
validate_shape(
?
save/Assign_35Assignlinear_user_featsave/RestoreV2:35*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
save/Assign_36Assignlinear_user_feat/Adamsave/RestoreV2:36*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
save/Assign_37Assignlinear_user_feat/Adam_1save/RestoreV2:37*
T0*#
_class
loc:@linear_user_feat* 
_output_shapes
:
Ã*
use_locking(*
validate_shape(
?
save/Assign_38Assignmlp/mlp_layer1/biassave/RestoreV2:38*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_39Assignmlp/mlp_layer1/bias/Adamsave/RestoreV2:39*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_40Assignmlp/mlp_layer1/bias/Adam_1save/RestoreV2:40*
T0*&
_class
loc:@mlp/mlp_layer1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_41Assignmlp/mlp_layer1/kernelsave/RestoreV2:41*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
use_locking(*
validate_shape(
?
save/Assign_42Assignmlp/mlp_layer1/kernel/Adamsave/RestoreV2:42*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
use_locking(*
validate_shape(
?
save/Assign_43Assignmlp/mlp_layer1/kernel/Adam_1save/RestoreV2:43*
T0*(
_class
loc:@mlp/mlp_layer1/kernel*
_output_shapes
:	P?*
use_locking(*
validate_shape(
?
save/Assign_44Assignmlp/mlp_layer2/biassave/RestoreV2:44*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_45Assignmlp/mlp_layer2/bias/Adamsave/RestoreV2:45*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_46Assignmlp/mlp_layer2/bias/Adam_1save/RestoreV2:46*
T0*&
_class
loc:@mlp/mlp_layer2/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_47Assignmlp/mlp_layer2/kernelsave/RestoreV2:47*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_48Assignmlp/mlp_layer2/kernel/Adamsave/RestoreV2:48*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_49Assignmlp/mlp_layer2/kernel/Adam_1save/RestoreV2:49*
T0*(
_class
loc:@mlp/mlp_layer2/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_50Assignmlp/mlp_layer3/biassave/RestoreV2:50*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_51Assignmlp/mlp_layer3/bias/Adamsave/RestoreV2:51*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_52Assignmlp/mlp_layer3/bias/Adam_1save/RestoreV2:52*
T0*&
_class
loc:@mlp/mlp_layer3/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_53Assignmlp/mlp_layer3/kernelsave/RestoreV2:53*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_54Assignmlp/mlp_layer3/kernel/Adamsave/RestoreV2:54*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/Assign_55Assignmlp/mlp_layer3/kernel/Adam_1save/RestoreV2:55*
T0*(
_class
loc:@mlp/mlp_layer3/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"??
cond_context????
?
mlp/dropout/cond/cond_textmlp/dropout/cond/pred_id:0mlp/dropout/cond/switch_t:0 *?
	mlp/Elu:0
mlp/dropout/cond/dropout/Cast:0
 mlp/dropout/cond/dropout/Const:0
)mlp/dropout/cond/dropout/GreaterEqual/y:0
'mlp/dropout/cond/dropout/GreaterEqual:0
%mlp/dropout/cond/dropout/Mul/Switch:1
mlp/dropout/cond/dropout/Mul:0
 mlp/dropout/cond/dropout/Mul_1:0
 mlp/dropout/cond/dropout/Shape:0
7mlp/dropout/cond/dropout/random_uniform/RandomUniform:0
mlp/dropout/cond/pred_id:0
mlp/dropout/cond/switch_t:08
mlp/dropout/cond/pred_id:0mlp/dropout/cond/pred_id:02
	mlp/Elu:0%mlp/dropout/cond/dropout/Mul/Switch:1
?
mlp/dropout/cond/cond_text_1mlp/dropout/cond/pred_id:0mlp/dropout/cond/switch_f:0*?
	mlp/Elu:0
"mlp/dropout/cond/Identity/Switch:0
mlp/dropout/cond/Identity:0
mlp/dropout/cond/pred_id:0
mlp/dropout/cond/switch_f:08
mlp/dropout/cond/pred_id:0mlp/dropout/cond/pred_id:0/
	mlp/Elu:0"mlp/dropout/cond/Identity/Switch:0
?
mlp/dropout_1/cond/cond_textmlp/dropout_1/cond/pred_id:0mlp/dropout_1/cond/switch_t:0 *?
mlp/Elu_1:0
!mlp/dropout_1/cond/dropout/Cast:0
"mlp/dropout_1/cond/dropout/Const:0
+mlp/dropout_1/cond/dropout/GreaterEqual/y:0
)mlp/dropout_1/cond/dropout/GreaterEqual:0
'mlp/dropout_1/cond/dropout/Mul/Switch:1
 mlp/dropout_1/cond/dropout/Mul:0
"mlp/dropout_1/cond/dropout/Mul_1:0
"mlp/dropout_1/cond/dropout/Shape:0
9mlp/dropout_1/cond/dropout/random_uniform/RandomUniform:0
mlp/dropout_1/cond/pred_id:0
mlp/dropout_1/cond/switch_t:0<
mlp/dropout_1/cond/pred_id:0mlp/dropout_1/cond/pred_id:06
mlp/Elu_1:0'mlp/dropout_1/cond/dropout/Mul/Switch:1
?
mlp/dropout_1/cond/cond_text_1mlp/dropout_1/cond/pred_id:0mlp/dropout_1/cond/switch_f:0*?
mlp/Elu_1:0
$mlp/dropout_1/cond/Identity/Switch:0
mlp/dropout_1/cond/Identity:0
mlp/dropout_1/cond/pred_id:0
mlp/dropout_1/cond/switch_f:0<
mlp/dropout_1/cond/pred_id:0mlp/dropout_1/cond/pred_id:03
mlp/Elu_1:0$mlp/dropout_1/cond/Identity/Switch:0
?
mlp/dropout_2/cond/cond_textmlp/dropout_2/cond/pred_id:0mlp/dropout_2/cond/switch_t:0 *?
mlp/Elu_2:0
!mlp/dropout_2/cond/dropout/Cast:0
"mlp/dropout_2/cond/dropout/Const:0
+mlp/dropout_2/cond/dropout/GreaterEqual/y:0
)mlp/dropout_2/cond/dropout/GreaterEqual:0
'mlp/dropout_2/cond/dropout/Mul/Switch:1
 mlp/dropout_2/cond/dropout/Mul:0
"mlp/dropout_2/cond/dropout/Mul_1:0
"mlp/dropout_2/cond/dropout/Shape:0
9mlp/dropout_2/cond/dropout/random_uniform/RandomUniform:0
mlp/dropout_2/cond/pred_id:0
mlp/dropout_2/cond/switch_t:0<
mlp/dropout_2/cond/pred_id:0mlp/dropout_2/cond/pred_id:06
mlp/Elu_2:0'mlp/dropout_2/cond/dropout/Mul/Switch:1
?
mlp/dropout_2/cond/cond_text_1mlp/dropout_2/cond/pred_id:0mlp/dropout_2/cond/switch_f:0*?
mlp/Elu_2:0
$mlp/dropout_2/cond/Identity/Switch:0
mlp/dropout_2/cond/Identity:0
mlp/dropout_2/cond/pred_id:0
mlp/dropout_2/cond/switch_f:0<
mlp/dropout_2/cond/pred_id:0mlp/dropout_2/cond/pred_id:03
mlp/Elu_2:0$mlp/dropout_2/cond/Identity/Switch:0
?
@mean_squared_error/assert_broadcastable/is_valid_shape/cond_text@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0 *?
3mean_squared_error/assert_broadcastable/is_scalar:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0x
3mean_squared_error/assert_broadcastable/is_scalar:0Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1?
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
?K
Bmean_squared_error/assert_broadcastable/is_valid_shape/cond_text_1@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_f:0*?#
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
omean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
jmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
emean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
Amean_squared_error/assert_broadcastable/is_valid_shape/switch_f:0
5mean_squared_error/assert_broadcastable/values/rank:0
6mean_squared_error/assert_broadcastable/values/shape:0
6mean_squared_error/assert_broadcastable/weights/rank:0
7mean_squared_error/assert_broadcastable/weights/shape:0?
6mean_squared_error/assert_broadcastable/weights/rank:0hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0?
5mean_squared_error/assert_broadcastable/values/rank:0fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0?
7mean_squared_error/assert_broadcastable/weights/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0?
6mean_squared_error/assert_broadcastable/values/shape:0umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0?
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:02?
?
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textZmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *?
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
omean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
jmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
emean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
6mean_squared_error/assert_broadcastable/values/shape:0
7mean_squared_error/assert_broadcastable/weights/shape:0?
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0?
7mean_squared_error/assert_broadcastable/weights/shape:0ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1?
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0?
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0?
6mean_squared_error/assert_broadcastable/values/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:12?
?
\mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*?
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0?
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0?
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
?
=mean_squared_error/assert_broadcastable/AssertGuard/cond_text=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0 *?
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency:0
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0~
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
?
?mean_squared_error/assert_broadcastable/AssertGuard/cond_text_1=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_f:0*?
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:0
Jmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1:0
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
>mean_squared_error/assert_broadcastable/AssertGuard/switch_f:0
3mean_squared_error/assert_broadcastable/is_scalar:0
>mean_squared_error/assert_broadcastable/is_valid_shape/Merge:0
6mean_squared_error/assert_broadcastable/values/shape:0
7mean_squared_error/assert_broadcastable/weights/shape:0?
>mean_squared_error/assert_broadcastable/is_valid_shape/Merge:0Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch:0?
7mean_squared_error/assert_broadcastable/weights/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1:0~
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
6mean_squared_error/assert_broadcastable/values/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2:0|
3mean_squared_error/assert_broadcastable/is_scalar:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3:0
?
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0?
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
?c
`mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*?.
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0?
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0?
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0?
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0?
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0?
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:02?&
?&
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textxmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *?#
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0?
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1?
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0?
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:12?
?
zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*?
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0?
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0?
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
?
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *?
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0?
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*?
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0?
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0?
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0?
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0?
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0"(
losses

mean_squared_error/value:0"
train_op

Adam"?
trainable_variables??
y
linear_user_feat:0linear_user_feat/Assignlinear_user_feat/read:02/linear_user_feat/Initializer/truncated_normal:08
y
linear_item_feat:0linear_item_feat/Assignlinear_item_feat/read:02/linear_item_feat/Initializer/truncated_normal:08
u
embed_user_feat:0embed_user_feat/Assignembed_user_feat/read:02.embed_user_feat/Initializer/truncated_normal:08
u
embed_item_feat:0embed_item_feat/Assignembed_item_feat/read:02.embed_item_feat/Initializer/truncated_normal:08
?
linear_sparse_feat:0linear_sparse_feat/Assignlinear_sparse_feat/read:021linear_sparse_feat/Initializer/truncated_normal:08
}
embed_sparse_feat:0embed_sparse_feat/Assignembed_sparse_feat/read:020embed_sparse_feat/Initializer/truncated_normal:08
}
linear_dense_feat:0linear_dense_feat/Assignlinear_dense_feat/read:020linear_dense_feat/Initializer/truncated_normal:08
y
embed_dense_feat:0embed_dense_feat/Assignembed_dense_feat/read:02/embed_dense_feat/Initializer/truncated_normal:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
?
mlp/mlp_layer1/kernel:0mlp/mlp_layer1/kernel/Assignmlp/mlp_layer1/kernel/read:022mlp/mlp_layer1/kernel/Initializer/random_uniform:08
z
mlp/mlp_layer1/bias:0mlp/mlp_layer1/bias/Assignmlp/mlp_layer1/bias/read:02'mlp/mlp_layer1/bias/Initializer/zeros:08
?
mlp/mlp_layer2/kernel:0mlp/mlp_layer2/kernel/Assignmlp/mlp_layer2/kernel/read:022mlp/mlp_layer2/kernel/Initializer/random_uniform:08
z
mlp/mlp_layer2/bias:0mlp/mlp_layer2/bias/Assignmlp/mlp_layer2/bias/read:02'mlp/mlp_layer2/bias/Initializer/zeros:08
?
mlp/mlp_layer3/kernel:0mlp/mlp_layer3/kernel/Assignmlp/mlp_layer3/kernel/read:022mlp/mlp_layer3/kernel/Initializer/random_uniform:08
z
mlp/mlp_layer3/bias:0mlp/mlp_layer3/bias/Assignmlp/mlp_layer3/bias/read:02'mlp/mlp_layer3/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08"?9
	variables?9?9
y
linear_user_feat:0linear_user_feat/Assignlinear_user_feat/read:02/linear_user_feat/Initializer/truncated_normal:08
y
linear_item_feat:0linear_item_feat/Assignlinear_item_feat/read:02/linear_item_feat/Initializer/truncated_normal:08
u
embed_user_feat:0embed_user_feat/Assignembed_user_feat/read:02.embed_user_feat/Initializer/truncated_normal:08
u
embed_item_feat:0embed_item_feat/Assignembed_item_feat/read:02.embed_item_feat/Initializer/truncated_normal:08
?
linear_sparse_feat:0linear_sparse_feat/Assignlinear_sparse_feat/read:021linear_sparse_feat/Initializer/truncated_normal:08
}
embed_sparse_feat:0embed_sparse_feat/Assignembed_sparse_feat/read:020embed_sparse_feat/Initializer/truncated_normal:08
}
linear_dense_feat:0linear_dense_feat/Assignlinear_dense_feat/read:020linear_dense_feat/Initializer/truncated_normal:08
y
embed_dense_feat:0embed_dense_feat/Assignembed_dense_feat/read:02/embed_dense_feat/Initializer/truncated_normal:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
?
mlp/mlp_layer1/kernel:0mlp/mlp_layer1/kernel/Assignmlp/mlp_layer1/kernel/read:022mlp/mlp_layer1/kernel/Initializer/random_uniform:08
z
mlp/mlp_layer1/bias:0mlp/mlp_layer1/bias/Assignmlp/mlp_layer1/bias/read:02'mlp/mlp_layer1/bias/Initializer/zeros:08
?
mlp/mlp_layer2/kernel:0mlp/mlp_layer2/kernel/Assignmlp/mlp_layer2/kernel/read:022mlp/mlp_layer2/kernel/Initializer/random_uniform:08
z
mlp/mlp_layer2/bias:0mlp/mlp_layer2/bias/Assignmlp/mlp_layer2/bias/read:02'mlp/mlp_layer2/bias/Initializer/zeros:08
?
mlp/mlp_layer3/kernel:0mlp/mlp_layer3/kernel/Assignmlp/mlp_layer3/kernel/read:022mlp/mlp_layer3/kernel/Initializer/random_uniform:08
z
mlp/mlp_layer3/bias:0mlp/mlp_layer3/bias/Assignmlp/mlp_layer3/bias/read:02'mlp/mlp_layer3/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
?
linear_user_feat/Adam:0linear_user_feat/Adam/Assignlinear_user_feat/Adam/read:02)linear_user_feat/Adam/Initializer/zeros:0
?
linear_user_feat/Adam_1:0linear_user_feat/Adam_1/Assignlinear_user_feat/Adam_1/read:02+linear_user_feat/Adam_1/Initializer/zeros:0
?
linear_item_feat/Adam:0linear_item_feat/Adam/Assignlinear_item_feat/Adam/read:02)linear_item_feat/Adam/Initializer/zeros:0
?
linear_item_feat/Adam_1:0linear_item_feat/Adam_1/Assignlinear_item_feat/Adam_1/read:02+linear_item_feat/Adam_1/Initializer/zeros:0
|
embed_user_feat/Adam:0embed_user_feat/Adam/Assignembed_user_feat/Adam/read:02(embed_user_feat/Adam/Initializer/zeros:0
?
embed_user_feat/Adam_1:0embed_user_feat/Adam_1/Assignembed_user_feat/Adam_1/read:02*embed_user_feat/Adam_1/Initializer/zeros:0
|
embed_item_feat/Adam:0embed_item_feat/Adam/Assignembed_item_feat/Adam/read:02(embed_item_feat/Adam/Initializer/zeros:0
?
embed_item_feat/Adam_1:0embed_item_feat/Adam_1/Assignembed_item_feat/Adam_1/read:02*embed_item_feat/Adam_1/Initializer/zeros:0
?
linear_sparse_feat/Adam:0linear_sparse_feat/Adam/Assignlinear_sparse_feat/Adam/read:02+linear_sparse_feat/Adam/Initializer/zeros:0
?
linear_sparse_feat/Adam_1:0 linear_sparse_feat/Adam_1/Assign linear_sparse_feat/Adam_1/read:02-linear_sparse_feat/Adam_1/Initializer/zeros:0
?
embed_sparse_feat/Adam:0embed_sparse_feat/Adam/Assignembed_sparse_feat/Adam/read:02*embed_sparse_feat/Adam/Initializer/zeros:0
?
embed_sparse_feat/Adam_1:0embed_sparse_feat/Adam_1/Assignembed_sparse_feat/Adam_1/read:02,embed_sparse_feat/Adam_1/Initializer/zeros:0
?
linear_dense_feat/Adam:0linear_dense_feat/Adam/Assignlinear_dense_feat/Adam/read:02*linear_dense_feat/Adam/Initializer/zeros:0
?
linear_dense_feat/Adam_1:0linear_dense_feat/Adam_1/Assignlinear_dense_feat/Adam_1/read:02,linear_dense_feat/Adam_1/Initializer/zeros:0
?
embed_dense_feat/Adam:0embed_dense_feat/Adam/Assignembed_dense_feat/Adam/read:02)embed_dense_feat/Adam/Initializer/zeros:0
?
embed_dense_feat/Adam_1:0embed_dense_feat/Adam_1/Assignembed_dense_feat/Adam_1/read:02+embed_dense_feat/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0
?
mlp/mlp_layer1/kernel/Adam:0!mlp/mlp_layer1/kernel/Adam/Assign!mlp/mlp_layer1/kernel/Adam/read:02.mlp/mlp_layer1/kernel/Adam/Initializer/zeros:0
?
mlp/mlp_layer1/kernel/Adam_1:0#mlp/mlp_layer1/kernel/Adam_1/Assign#mlp/mlp_layer1/kernel/Adam_1/read:020mlp/mlp_layer1/kernel/Adam_1/Initializer/zeros:0
?
mlp/mlp_layer1/bias/Adam:0mlp/mlp_layer1/bias/Adam/Assignmlp/mlp_layer1/bias/Adam/read:02,mlp/mlp_layer1/bias/Adam/Initializer/zeros:0
?
mlp/mlp_layer1/bias/Adam_1:0!mlp/mlp_layer1/bias/Adam_1/Assign!mlp/mlp_layer1/bias/Adam_1/read:02.mlp/mlp_layer1/bias/Adam_1/Initializer/zeros:0
?
mlp/mlp_layer2/kernel/Adam:0!mlp/mlp_layer2/kernel/Adam/Assign!mlp/mlp_layer2/kernel/Adam/read:02.mlp/mlp_layer2/kernel/Adam/Initializer/zeros:0
?
mlp/mlp_layer2/kernel/Adam_1:0#mlp/mlp_layer2/kernel/Adam_1/Assign#mlp/mlp_layer2/kernel/Adam_1/read:020mlp/mlp_layer2/kernel/Adam_1/Initializer/zeros:0
?
mlp/mlp_layer2/bias/Adam:0mlp/mlp_layer2/bias/Adam/Assignmlp/mlp_layer2/bias/Adam/read:02,mlp/mlp_layer2/bias/Adam/Initializer/zeros:0
?
mlp/mlp_layer2/bias/Adam_1:0!mlp/mlp_layer2/bias/Adam_1/Assign!mlp/mlp_layer2/bias/Adam_1/read:02.mlp/mlp_layer2/bias/Adam_1/Initializer/zeros:0
?
mlp/mlp_layer3/kernel/Adam:0!mlp/mlp_layer3/kernel/Adam/Assign!mlp/mlp_layer3/kernel/Adam/read:02.mlp/mlp_layer3/kernel/Adam/Initializer/zeros:0
?
mlp/mlp_layer3/kernel/Adam_1:0#mlp/mlp_layer3/kernel/Adam_1/Assign#mlp/mlp_layer3/kernel/Adam_1/read:020mlp/mlp_layer3/kernel/Adam_1/Initializer/zeros:0
?
mlp/mlp_layer3/bias/Adam:0mlp/mlp_layer3/bias/Adam/Assignmlp/mlp_layer3/bias/Adam/read:02,mlp/mlp_layer3/bias/Adam/Initializer/zeros:0
?
mlp/mlp_layer3/bias/Adam_1:0!mlp/mlp_layer3/bias/Adam_1/Assign!mlp/mlp_layer3/bias/Adam_1/read:02.mlp/mlp_layer3/bias/Adam_1/Initializer/zeros:0
x
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assigndense_1/kernel/Adam/read:02'dense_1/kernel/Adam/Initializer/zeros:0
?
dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assigndense_1/kernel/Adam_1/read:02)dense_1/kernel/Adam_1/Initializer/zeros:0
p
dense_1/bias/Adam:0dense_1/bias/Adam/Assigndense_1/bias/Adam/read:02%dense_1/bias/Adam/Initializer/zeros:0
x
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assigndense_1/bias/Adam_1/read:02'dense_1/bias/Adam_1/Initializer/zeros:0*?
predict?
6
dense_values&
Placeholder_4:0?????????
2
item_indices"
Placeholder_2:0?????????
8
sparse_indices&
Placeholder_3:0?????????
2
user_indices"
Placeholder_1:0?????????
logits
	Squeeze:0tensorflow/serving/predict