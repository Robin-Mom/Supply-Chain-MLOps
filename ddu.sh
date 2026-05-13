#!/bin/bash

# =============================================
# Script de diagnostic de consommation disque (en Mo)
# Usage: ./ddu.sh [étape]
# =============================================

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Fonctions d'affichage
print_title() { echo -e "${BLUE}=== $1 ===${NC}"; }
print_metric() { echo -e "${YELLOW}$1:${NC} ${2:-0} Mo"; }
print_section() { echo -e "\n${GREEN}$1${NC}\n----------------------------------------"; }

# Fonction pour obtenir la taille en Mo (0 si erreur)
get_size_mb() {
    local path=$1
    if [ -d "$path" ]; then
        du -sm "$path" 2>/dev/null | awk '{print $1}'
    else
        echo "0"
    fi
}

# Fonction pour obtenir la taille Docker en Mo
get_docker_size_mb() {
    local metric=$1
    docker system df --format "{{.$metric}}" 2>/dev/null | tail -n 1 | awk '{
        if ($1 ~ /G/) { printf "%.0f", $1 * 1024 }
        else if ($1 ~ /M/) { gsub(/M/, "", $1); print $1 + 0 }
        else { print "0" }
    }'
}

# Fonction pour obtenir l'espace disque en Mo
get_disk_usage_mb() { df -m / | awk 'NR==2 {print $3}'; }
get_disk_available_mb() { df -m / | awk 'NR==2 {print $4}'; }

# Fonction pour calculer la taille des __pycache__ et .pyc
get_pycache_size_mb() {
    local size=0
    size=$((size + $(find . -type d -name "__pycache__" -exec du -sm {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}') ))
    size=$((size + $(find . -type f -name "*.pyc" -exec du -sm {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}') ))
    echo "${size:-0}"
}

# Afficher toutes les métriques
display_all_metrics() {
    print_title "DIAGNOSTIC D'ESPACE DISQUE (en Mo) - $(date)"

    print_section "💾 Espace disque global"
    print_metric "Espace disque total utilisé" "$(get_disk_usage_mb)"
    print_metric "Espace disponible" "$(get_disk_available_mb)"

    print_section "🐳 Docker"
    local images_size=$(get_docker_size_mb "Images")
    local containers_size=$(get_docker_size_mb "Containers")
    local buildkit_size=$(get_docker_size_mb "BuildCache")
    local volumes_size=$(get_docker_size_mb "Volumes")
    print_metric "Taille des images Docker" "$images_size"
    print_metric "Taille des conteneurs Docker" "$containers_size"
    print_metric "Cache BuildKit" "$buildkit_size"
    print_metric "Volumes Docker" "$volumes_size"
    local docker_total=$((images_size + containers_size + buildkit_size + volumes_size))
    print_metric "Total Docker (estimé)" "$docker_total"

    print_section "🐍 uv et Python"
    local uv_cache_size=$(get_size_mb ~/.cache/uv)
    local uv_custom_size=$(get_size_mb "${UV_CACHE_DIR:-/dev/null}")
    local venv_size=$(get_size_mb .venv)
    local uv_project_size=$(get_size_mb .uv_cache)
    print_metric "Cache uv local (~/.cache/uv)" "$uv_cache_size"
    print_metric "Cache uv personnalisé (UV_CACHE_DIR)" "$uv_custom_size"
    print_metric "Dossier .venv local" "$venv_size"
    print_metric "Dossier .uv_cache (projet)" "$uv_project_size"
    local uv_total=$((uv_cache_size + uv_custom_size + uv_project_size))
    print_metric "Total uv (estimé)" "$uv_total"

    print_section "📁 Projet local"
    local project_size=$(get_size_mb .)
    local node_modules_size=$(get_size_mb node_modules)
    local pycache_size=$(get_pycache_size_mb)
    print_metric "Taille du projet (.)" "$project_size"
    print_metric "Dossier node_modules" "$node_modules_size"
    print_metric "Dossier __pycache__ + .pyc" "$pycache_size"

    print_section "📊 Résumé des ressources clés"
    local grand_total=$((docker_total + uv_total + project_size + node_modules_size + pycache_size))
    print_metric "Total estimé (Docker + uv + projet)" "$grand_total"
}

# Sauvegarder les métriques
save_metrics() {
    local step=$1
    local filename="disk_usage_${step}_$(date +%Y%m%d_%H%M%S).log"
    echo "Sauvegarde des métriques dans $filename..."
    { echo "=== DIAGNOSTIC À L'ÉTAPE: $step ==="; echo "Date: $(date)"; echo ""; display_all_metrics; } > "$filename" 2>&1
    echo -e "${GREEN}Métriques sauvegardées dans $filename${NC}"
}

# Comparer deux étapes
compare_steps() {
    local step1=$1 step2=$2
    local file1="disk_usage_${step1}_*.log"
    local file2="disk_usage_${step2}_*.log"

    print_title "COMPARAISON ENTRE $step1 ET $step2 (en Mo)"

    local latest_file1=$(ls -t $file1 2>/dev/null | head -n 1)
    local latest_file2=$(ls -t $file2 2>/dev/null | head -n 1)

    if [ -z "$latest_file1" ] || [ -z "$latest_file2" ]; then
        echo -e "${RED}Erreur: Fichiers de métriques introuvables pour $step1 ou $step2${NC}"
        return 1
    fi

    echo -e "${YELLOW}Fichiers utilisés:${NC}"
    echo "  - $step1: $latest_file1"
    echo "  - $step2: $latest_file2"
    echo ""

    print_section "📉 Différences"

    # Fonction pour extraire une métrique d'un fichier
    extract_metric() {
        local file=$1 metric=$2
        grep "$metric" "$file" | awk '{print $NF}' | tr -d 'Mo'
    }

    # Calculer les différences
    local disk_before=$(extract_metric "$latest_file1" "Espace disque total utilisé")
    local disk_after=$(extract_metric "$latest_file2" "Espace disque total utilisé")
    local disk_diff=$((disk_after - disk_before))
    print_metric "Δ Espace disque total" "$disk_diff"

    local images_before=$(extract_metric "$latest_file1" "Taille des images Docker")
    local images_after=$(extract_metric "$latest_file2" "Taille des images Docker")
    local images_diff=$((images_after - images_before))
    print_metric "Δ Taille des images Docker" "$images_diff"

    local buildkit_before=$(extract_metric "$latest_file1" "Cache BuildKit")
    local buildkit_after=$(extract_metric "$latest_file2" "Cache BuildKit")
    local buildkit_diff=$((buildkit_after - buildkit_before))
    print_metric "Δ Cache BuildKit" "$buildkit_diff"

    local uv_before=$(extract_metric "$latest_file1" "Cache uv local")
    local uv_after=$(extract_metric "$latest_file2" "Cache uv local")
    local uv_diff=$((uv_after - uv_before))
    print_metric "Δ Cache uv local" "$uv_diff"

    local project_before=$(extract_metric "$latest_file1" "Taille du projet")
    local project_after=$(extract_metric "$latest_file2" "Taille du projet")
    local project_diff=$((project_after - project_before))
    print_metric "Δ Taille du projet" "$project_diff"
}

# =============================================
# EXÉCUTION PRINCIPALE
# =============================================
case "${1:-}" in
    "before_uv")
        print_title "DIAGNOSTIC AVANT uv sync (en Mo)"
        display_all_metrics
        save_metrics "before_uv"
        ;;
    "after_uv")
        print_title "DIAGNOSTIC APRÈS uv sync (en Mo)"
        display_all_metrics
        save_metrics "after_uv"
        ;;
    "before_docker")
        print_title "DIAGNOSTIC AVANT docker build (en Mo)"
        display_all_metrics
        save_metrics "before_docker"
        ;;
    "after_docker")
        print_title "DIAGNOSTIC APRÈS docker build (en Mo)"
        display_all_metrics
        save_metrics "after_docker"
        ;;
    "full")
        print_title "DIAGNOSTIC COMPLET (en Mo)"
        display_all_metrics
        save_metrics "full_$(date +%Y%m%d_%H%M%S)"
        ;;
    "compare")
        if [ $# -lt 3 ]; then
            echo -e "${RED}Usage: $0 compare <étape1> <étape2>${NC}"
            exit 1
        fi
        compare_steps "$2" "$3"
        ;;
    "")
        display_all_metrics
        ;;
    *)
        echo -e "${RED}Étape inconnue: $1${NC}"
        echo "Étapes valides: before_uv, after_uv, before_docker, after_docker, full, compare"
        exit 1
        ;;
esac