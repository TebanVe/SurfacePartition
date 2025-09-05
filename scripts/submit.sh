#!/bin/bash

# Generic submission script for UPPMAX (Rackham) to run surface partition optimizations
# Usage examples:
#   bash scripts/submit.sh --input parameters/input.yaml
#   bash scripts/submit.sh --input parameters/input.yaml --surface ring --time 24:00:00 --venv ringtest-3.9
#   bash scripts/submit.sh --input parameters/input.yaml --solution-dir /proj/snic2020-15-36/private/RING/SOLUTIONS

set -euo pipefail

# -------------------------------
# Project configuration (edit as needed)
# -------------------------------
PROJECT_FOLDER="snic2020-15-36"   # Project folder for file paths on UPPMAX (/proj/<folder>)
PROJECT_ID="uppmax2025-2-192"     # SLURM account (-A)
PROJECT_BASE="/proj/${PROJECT_FOLDER}"

# -------------------------------
# Defaults
# -------------------------------
INPUT_FILE="parameters/input.yaml"
OUTPUT_DIR="results"
SOLUTION_DIR="${PROJECT_BASE}/private/LINKED_LST_MANIFOLD/OPTIM_SOLUTIONS"
TIME_LIMIT="12:00:00"
VENV_NAME="partition"        # Virtual environment name under $HOME
SURFACE_ARG=""               # Optional override for surface (falls back to YAML)

# -------------------------------
# Parse command line arguments
# -------------------------------
while [[ $# -gt 0 ]]; do
	case $1 in
		--input)
			INPUT_FILE="$2"; shift 2;;
		--output)
			OUTPUT_DIR="$2"; shift 2;;
		--solution-dir)
			SOLUTION_DIR="$2"; shift 2;;
		--time)
			TIME_LIMIT="$2"; shift 2;;
		--venv)
			VENV_NAME="$2"; shift 2;;
		--surface)
			SURFACE_ARG="$2"; shift 2;;
		*)
			echo "Unknown option: $1"; exit 1;;
	esac
done

# -------------------------------
# Helpers
# -------------------------------
abspath() { python3 - "$1" << 'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}
extract_yaml() {
	# Extract a simple "key: value" from YAML (first match), stripping quotes
	local key="$1"; local file="$2"
	grep -E "^[[:space:]]*${key}:[[:space:]]*" "$file" | head -n1 | awk -F': ' '{print $2}' | tr -d '"' || true
}

# Resolve absolute paths and repo root
REPO_ROOT="$(pwd)"
INPUT_FILE_ABS="$(abspath "$INPUT_FILE")"
OUTPUT_DIR_ABS="$(abspath "$OUTPUT_DIR")"
SOLUTION_DIR_ABS="$(abspath "$SOLUTION_DIR")"

mkdir -p "$OUTPUT_DIR_ABS"
mkdir -p "$SOLUTION_DIR_ABS"

# -------------------------------
# Read YAML for job naming (robust to ring/torus naming variants)
# -------------------------------
if [[ ! -f "$INPUT_FILE_ABS" ]]; then
	echo "Error: Input file not found: $INPUT_FILE_ABS"; exit 1
fi

# Surface from YAML (optional)
SURFACE_YAML="$(extract_yaml surface "$INPUT_FILE_ABS")"
if [[ -n "$SURFACE_ARG" ]]; then
	SURFACE="$SURFACE_ARG"
else
	SURFACE="${SURFACE_YAML:-ring}"
fi

# Common parameters
N_PARTITIONS="$(extract_yaml n_partitions "$INPUT_FILE_ABS")"; N_PARTITIONS=${N_PARTITIONS:-3}
LAMBDA="$(extract_yaml lambda_penalty "$INPUT_FILE_ABS")"; LAMBDA=${LAMBDA:-0.0}
SEED="$(extract_yaml seed "$INPUT_FILE_ABS")"; SEED=${SEED:-42}
REF_LEVELS="$(extract_yaml refinement_levels "$INPUT_FILE_ABS")"; REF_LEVELS=${REF_LEVELS:-1}

# Ring-style keys (nr/na)
NR="$(extract_yaml n_radial "$INPUT_FILE_ABS")"; NA="$(extract_yaml n_angular "$INPUT_FILE_ABS")"
NR_INC="$(extract_yaml n_radial_increment "$INPUT_FILE_ABS")"; NA_INC="$(extract_yaml n_angular_increment "$INPUT_FILE_ABS")"

# Torus-style keys (nt/np)
NT="$(extract_yaml n_theta "$INPUT_FILE_ABS")"; NP="$(extract_yaml n_phi "$INPUT_FILE_ABS")"
NT_INC="$(extract_yaml n_theta_increment "$INPUT_FILE_ABS")"; NP_INC="$(extract_yaml n_phi_increment "$INPUT_FILE_ABS")"

# Choose labels and values based on available keys
if [[ -n "$NR" && -n "$NA" ]]; then
	V1_LABEL="nr"; V2_LABEL="na"; V1="$NR"; V2="$NA"; V1_INC="${NR_INC:-0}"; V2_INC="${NA_INC:-0}"
elif [[ -n "$NT" && -n "$NP" ]]; then
	V1_LABEL="nt"; V2_LABEL="np"; V1="$NT"; V2="$NP"; V1_INC="${NT_INC:-0}"; V2_INC="${NP_INC:-0}"
else
	# Fallback labels
	V1_LABEL="v1"; V2_LABEL="v2"; V1="0"; V2="0"; V1_INC="0"; V2_INC="0"
fi

# Compute final resolutions if refining
if [[ "$REF_LEVELS" -gt 1 ]]; then
	FINAL_V1=$(( V1 + (REF_LEVELS - 1) * V1_INC ))
	FINAL_V2=$(( V2 + (REF_LEVELS - 1) * V2_INC ))
	V1_INFO="${V1}-${FINAL_V1}_inc${V1_INC}"
	V2_INFO="${V2}-${FINAL_V2}_inc${V2_INC}"
else
	V1_INFO="${V1}"; V2_INFO="${V2}"
fi

# -------------------------------
# SLURM job metadata
# -------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="${TIMESTAMP}_surf${SURFACE}_npart${N_PARTITIONS}_v1${V1_LABEL}${V1_INFO}_v2${V2_LABEL}${V2_INFO}_lam${LAMBDA}_seed${SEED}"
JOB_LOGS_DIR="${OUTPUT_DIR_ABS}/job_logs/${JOB_NAME}"
mkdir -p "$JOB_LOGS_DIR"

# -------------------------------
# Create temporary SLURM script
# -------------------------------
SLURM_SCRIPT=$(mktemp)
cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH -A ${PROJECT_ID}
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t ${TIME_LIMIT}
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${JOB_LOGS_DIR}/${JOB_NAME}.out
#SBATCH -e ${JOB_LOGS_DIR}/${JOB_NAME}.err

module load python/3.9.5

# Activate virtual environment
if [ -d "\$HOME/${VENV_NAME}" ]; then
	echo "Activating virtual environment: ${VENV_NAME}"
	source "\$HOME/${VENV_NAME}/bin/activate"
else
	echo "Error: Virtual environment '${VENV_NAME}' not found in \$HOME"; exit 1
fi

# Ensure paths exist
mkdir -p "${OUTPUT_DIR_ABS}"
mkdir -p "${SOLUTION_DIR_ABS}"

# Environment
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}/src:\$PYTHONPATH"
cd "${REPO_ROOT}"

# Run orchestrator (surface-agnostic)
python examples/find_surface_partition.py \
	--input "${INPUT_FILE_ABS}" \
	--solution-dir "${SOLUTION_DIR_ABS}" \
	--surface "${SURFACE}"
EOF

# -------------------------------
# Submit and cleanup
# -------------------------------
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
rm -f "$SLURM_SCRIPT"

echo "Job submitted: ${JOB_NAME} (ID: ${JOB_ID})"
echo "Logs: ${JOB_LOGS_DIR}/${JOB_NAME}.out (and .err)" 