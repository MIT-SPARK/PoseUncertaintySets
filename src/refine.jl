using JuMP
using Ipopt

function local_refine_tssos(opt, data; QUIET=false, tol=1e-4, startpoint=[])

## CONVERT DATA
n = data.n
nb = data.nb
numeq = data.numeq
if typeof(data) == TSSOS.cpop_data && !isempty(data.gb)
    m = length(data.pop)-1
    supp = Vector{Array{UInt8,2}}(undef, m+1)
    coe = Vector{Vector{Float64}}(undef, m+1)
    supp[2:m+1-numeq] = data.supp[2:end]
    coe[2:m+1-numeq] = data.coe[2:end]
    for k in [1; [k for k=m+2-numeq:m+1]]
        mons = MultivariatePolynomials.monomials(data.pop[k])
        coe[k] = MultivariatePolynomials.coefficients(data.pop[k])
        supp[k] = zeros(UInt8, n, length(mons))
        for i in eachindex(mons), j = 1:n
            @inbounds supp[k][j,i] = MultivariatePolynomials.degree(mons[i], data.x[j])
        end
    end
else
    m = data.m
    supp = data.supp
    coe = data.coe
end
if !isempty(startpoint)
    for i = 1:n
        if abs(startpoint[i]) < 1e-10
            startpoint[i] = 1e-10
        end
    end
end

## LOCAL SOLUTION
model = Model(optimizer_with_attributes(Ipopt.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), QUIET)
# set_optimizer_attribute(model, "max_iter", 500)
if QUIET == true
    set_optimizer_attribute(model, "print_level", 0)
end
if isempty(startpoint)
    @variable(model, x[1:n], start = 0)
else
    @variable(model, x[i=1:n], start = startpoint[i])
end
@NLobjective(model, Min, sum(coe[1][j]*prod(x[supp[1][j][k]] for k=1:length(supp[1][j])) for j=1:length(supp[1])))
for i = 1:nb
    @NLconstraint(model, x[i]^2-1==0)
end
for i in 1:m-numeq
    @NLconstraint(model, sum(coe[i+1][j]*prod(x[supp[i+1][j][k]] for k=1:length(supp[i+1][j])) for j=1:length(supp[i+1]))>=0)
end
for i in m-numeq+1:m
    @NLconstraint(model, sum(coe[i+1][j]*prod(x[supp[i+1][j][k]] for k=1:length(supp[i+1][j])) for j=1:length(supp[i+1]))==0)
end
optimize!(model)
status = termination_status(model)
objv = objective_value(model)
if QUIET == false
    println("optimum = $objv")
end
ub = objv
rsol = value.(x)

## STATUS
if status == MOI.LOCALLY_SOLVED
    gap = abs(opt-ub)/max(1, abs(ub))
    if gap < tol
        @printf "Global optimality certified with relative optimality gap %.6f%%!\n" 100*gap
    else
        @printf "Found a locally optimal solution by Ipopt, giving an upper bound: %.8f.\nThe relative optimality gap is: %.6f%%.\n" ub 100*gap
    end
else
    gap = 1e6
end

# rsol: locally refined solution

return rsol, status, gap


end