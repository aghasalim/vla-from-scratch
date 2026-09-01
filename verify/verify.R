# The claims the README makes in words, tested in base R.
#
# scripts/check_numbers.py says so itself: it looks for quoted figures and
# "does not check claims written in words". The load bearing sentences in this
# repository are exactly those. "The seed ranges do not overlap at all" is the
# result; the numbers either side of it are only how it is evidenced. Nothing
# checked the comparison, only the operands.
#
# So this reads results/heads.csv and results/step-sweep.csv, recomputes the
# ranges with base R, tests each stated relation, and adds the exact permutation
# test that three seeds allow: with 3 against 3 there are choose(6,3) = 20 label
# assignments, which come in mirrored pairs, so the smallest two sided p value
# attainable is 2/20 = 0.10. A perfect separation is worth exactly that and no
# more, and no amount of gap between the two groups buys anything below it.
#
#   Rscript verify/verify.R

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
heads <- read.csv(file.path(root, "results", "heads.csv"), check.names = FALSE)
sweep <- read.csv(file.path(root, "results", "step-sweep.csv"), check.names = FALSE)

fails <- character(0)
checks <- 0L
check <- function(ok, msg) {
  checks <<- checks + 1L
  if (!isTRUE(ok)) fails <<- c(fails, msg)
}

rng <- function(df, head, col) {
  v <- df[df$head == head, col]
  stopifnot(length(v) > 0)
  c(min(v), max(v))
}
disjoint <- function(a, b) a[2] < b[1] || b[2] < a[1]
fmt <- function(v, d = 3) formatC(v, format = "f", digits = d)

REG <- "regression"
MULTI <- c("discrete bins", "diffusion", "flow (pi-0 style)")

# 1. "The seed ranges do not overlap at all: regression spans 8.20 to 9.45, and
#    the widest multimodal range is 1.43 to 4.37."
reg_b <- rng(heads, REG, "blocked_steps")
mm_b <- c(min(heads$blocked_steps[heads$head != REG]),
          max(heads$blocked_steps[heads$head != REG]))
check(identical(fmt(reg_b, 2), c("8.20", "9.45")),
      paste("regression collision span is", paste(fmt(reg_b, 2), collapse = " to ")))
check(identical(fmt(mm_b, 2), c("1.43", "4.37")),
      paste("multimodal collision envelope is", paste(fmt(mm_b, 2), collapse = " to ")))
check(disjoint(reg_b, mm_b), "regression and multimodal collision ranges overlap")
for (h in MULTI) {
  check(disjoint(reg_b, rng(heads, h, "blocked_steps")),
        paste("regression collision range overlaps", h))
}

# 2. "Flow spans 0.270 to 0.297 against regression's 0.211 to 0.223, which do
#    not overlap." And discrete bins and diffusion do overlap regression.
reg_s <- rng(heads, REG, "success")
flow_s <- rng(heads, "flow (pi-0 style)", "success")
check(identical(fmt(reg_s), c("0.211", "0.223")),
      paste("regression success span is", paste(fmt(reg_s), collapse = " to ")))
check(identical(fmt(flow_s), c("0.270", "0.297")),
      paste("flow success span is", paste(fmt(flow_s), collapse = " to ")))
check(disjoint(reg_s, flow_s), "flow and regression success ranges overlap")
for (h in c("discrete bins", "diffusion")) {
  check(!disjoint(reg_s, rng(heads, h, "success")),
        paste(h, "success range no longer overlaps regression's"))
}

# 3. The step sweep sentence: flow at 1 step has the highest median, its three
#    seeds are 0.230, 0.309 and 0.316, the 0.230 is the worst of any flow row,
#    and its span covers every other median in the table except diffusion at 10.
key <- paste(sweep$head, sweep$steps)
med <- tapply(sweep$success, key, median)
f1 <- sort(sweep$success[sweep$head == "flow (pi-0 style)" & sweep$steps == 1])
check(identical(fmt(f1), c("0.230", "0.309", "0.316")),
      paste("flow at 1 step seeds are", paste(fmt(f1), collapse = ", ")))
check(max(med) == med[["flow (pi-0 style) 1"]], "flow at 1 step no longer has the highest median")
check(min(sweep$success[sweep$head == "flow (pi-0 style)"]) == f1[1],
      "0.230 is no longer the worst success any flow row records")
others <- med[names(med) != "flow (pi-0 style) 1"]
outside <- names(others)[others < f1[1] | others > f1[3]]
check(identical(outside, "diffusion 10"),
      paste("medians outside the flow at 1 step span are now",
            if (length(outside) == 0) "none" else paste(outside, collapse = ", ")))

# 4. The exact permutation test. Not a claim in the README, which is the point:
#    the separation is stated as a fact about ranges and never given a p value.
perm_p <- function(a, b) {
  pool <- c(a, b)
  idx <- combn(length(pool), length(a))
  stat <- abs(mean(a) - mean(b))
  draws <- apply(idx, 2, function(i) abs(mean(pool[i]) - mean(pool[-i])))
  mean(draws >= stat - 1e-12)
}
cat("verify.R: exact permutation tests, 3 seeds against 3, 20 label assignments\n")
for (h in MULTI) {
  pb <- perm_p(heads$blocked_steps[heads$head == REG], heads$blocked_steps[heads$head == h])
  ps <- perm_p(heads$success[heads$head == REG], heads$success[heads$head == h])
  cat(sprintf("  regression vs %-18s collisions p = %.2f   success p = %.2f\n", h, pb, ps))
  check(pb <= 0.10, paste("collision separation from", h, "is no longer at the 0.10 floor"))
}

cat(sprintf("verify.R: %d stated relations tested, %d failed\n", checks, length(fails)))
for (f in fails) cat("  -", f, "\n")
if (length(fails) > 0) quit(status = 1)
