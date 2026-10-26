#!/usr/bin/env bash
set -x
cd /workspace/project

echo "[$(date)] Starting EXP 1" | tee -a logs/exp_all.log
python -u -m exp.exp_1_primary > logs/exp1.log 2>&1
echo "[$(date)] EXP 1 exit=$?" | tee -a logs/exp_all.log

echo "[$(date)] Starting EXP 2" | tee -a logs/exp_all.log
python -u -m exp.exp_2_order_selection > logs/exp2.log 2>&1
echo "[$(date)] EXP 2 exit=$?" | tee -a logs/exp_all.log

echo "[$(date)] Starting EXP 3" | tee -a logs/exp_all.log
python -u -m exp.exp_3_template_tracking > logs/exp3.log 2>&1
echo "[$(date)] EXP 3 exit=$?" | tee -a logs/exp_all.log

echo "[$(date)] Starting EXP 4" | tee -a logs/exp_all.log
python -u -m exp.exp_4_benchmarks > logs/exp4.log 2>&1
echo "[$(date)] EXP 4 exit=$?" | tee -a logs/exp_all.log

echo "[$(date)] All done" | tee -a logs/exp_all.log
