#!/bin/bash
# GCP TPU VM Setup Script for MSSA Training
# Usage: bash scripts/setup_gcp.sh

set -e

# ============================================
# Configuration - Update these values
# ============================================
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
TPU_NAME="${TPU_NAME:-mssa-training}"
TPU_TYPE="${TPU_TYPE:-v3-8}"

# ============================================
# Colors for output
# ============================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================
# Check prerequisites
# ============================================
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
        log_error "Not authenticated. Run: gcloud auth login"
        exit 1
    fi
    
    log_info "Prerequisites OK"
}

# ============================================
# Create TPU VM
# ============================================
create_tpu_vm() {
    log_info "Creating TPU VM: $TPU_NAME (type: $TPU_TYPE) in $ZONE..."
    
    # Check if TPU already exists
    if gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE --project=$PROJECT_ID &> /dev/null; then
        log_warn "TPU VM '$TPU_NAME' already exists. Skipping creation."
        return 0
    fi
    
    gcloud compute tpus tpu-vm create $TPU_NAME \
        --zone=$ZONE \
        --project=$PROJECT_ID \
        --accelerator-type=$TPU_TYPE \
        --version=tpu-ubuntu2204-base
    
    log_info "TPU VM created successfully!"
}

# ============================================
# Setup VM environment
# ============================================
setup_vm_environment() {
    log_info "Setting up VM environment..."
    
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --command="
        set -e
        
        echo 'Updating system packages...'
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-pip python3-venv git htop tmux
        
        echo 'Creating virtual environment...'
        python3 -m venv ~/mssa-env
        source ~/mssa-env/bin/activate
        
        echo 'Upgrading pip...'
        pip install --upgrade pip -q
        
        echo 'Environment setup complete!'
    "
    
    log_info "VM environment setup complete!"
}

# ============================================
# Clone and setup project
# ============================================
setup_project() {
    local REPO_URL="$1"
    local BRANCH="${2:-curriculum-jax}"
    
    if [ -z "$REPO_URL" ]; then
        log_warn "No repo URL provided. Skipping project clone."
        return 0
    fi
    
    log_info "Cloning project from $REPO_URL (branch: $BRANCH)..."
    
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --command="
        source ~/mssa-env/bin/activate
        
        if [ -d ~/MSSA ]; then
            echo 'Project already exists. Pulling latest changes...'
            cd ~/MSSA
            git fetch origin
            git checkout $BRANCH
            git pull origin $BRANCH
        else
            echo 'Cloning project...'
            git clone $REPO_URL ~/MSSA
            cd ~/MSSA
            git checkout $BRANCH
        fi
        
        echo 'Installing dependencies...'
        pip install -r requirements_cpu.txt -q
        
        echo 'Project setup complete!'
    "
    
    log_info "Project setup complete!"
}

# ============================================
# Install JAX with TPU support
# ============================================
install_jax_tpu() {
    log_info "Installing JAX with TPU support..."
    
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --command="
        source ~/mssa-env/bin/activate
        
        echo 'Installing JAX for TPU...'
        pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
        pip install flax optax distrax chex -q
        
        echo 'Verifying JAX TPU installation...'
        python3 -c \"
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('TPU cores available:', len([d for d in jax.devices() if 'TPU' in str(d)]))
\"
        
        echo 'JAX TPU installation complete!'
    "
    
    log_info "JAX TPU installation complete!"
}

# ============================================
# Print connection info
# ============================================
print_connection_info() {
    echo ""
    echo "=========================================="
    echo "  TPU VM Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Connect to your VM:"
    echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID"
    echo ""
    echo "Start training (CPU):"
    echo "  source ~/mssa-env/bin/activate"
    echo "  cd ~/MSSA"
    echo "  bash scripts/run_cpu_training.sh"
    echo ""
    echo "Start training (JAX/TPU):"
    echo "  source ~/mssa-env/bin/activate"
    echo "  cd ~/MSSA"
    echo "  python train_jax/train_curriculum.py"
    echo ""
    echo "=========================================="
}

# ============================================
# Main
# ============================================
main() {
    echo "=========================================="
    echo "  MSSA GCP TPU VM Setup"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    
    # Parse arguments
    REPO_URL=""
    BRANCH="curriculum-jax"
    SKIP_JAX=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --repo)
                REPO_URL="$2"
                shift 2
                ;;
            --branch)
                BRANCH="$2"
                shift 2
                ;;
            --project)
                PROJECT_ID="$2"
                shift 2
                ;;
            --zone)
                ZONE="$2"
                shift 2
                ;;
            --name)
                TPU_NAME="$2"
                shift 2
                ;;
            --skip-jax)
                SKIP_JAX=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    create_tpu_vm
    setup_vm_environment
    setup_project "$REPO_URL" "$BRANCH"
    
    if [ "$SKIP_JAX" = false ]; then
        install_jax_tpu
    fi
    
    print_connection_info
}

main "$@"
