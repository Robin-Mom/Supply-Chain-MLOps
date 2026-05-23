#!/bin/bash
# secretgen.sh
# POUR NE PAS TRANSMETTRE DE CLEF ET DE SECRET ENTRE DEV
# pour éviter de travailler avec aws ou git-crypt quand on a pas le temps
# ET QUAND ON EST PAS SUR UN SERVEUR CENTRAL NI AVEC UNE AUTORITE DE CERTIFICATION
# ALORS ON REGENERE CHEZ CHACUN EN ENV DE DEV AVEC CERTIFICAT LOCAL

# exemple de lancement depuis ./bin
# secretgen localhost ../certs


#Terminal color
export GREEN=$(tput setaf 2)
export RED=$(tput setaf 1)
export BLUE=$(tput setaf 4)
export RESET=$(tput sgr0)

function title(){
  echo -e "\n${BLUE}$1${RESET}"
}

function result(){
  retVal="$1"
  test="$2"
  if [[ $retVal != 0 ]]; then
    echo "$RED ❌ Error :$retVal $test not passed $RESET"
  else
    echo "$GREEN ✅ Success :$retVal $test passed $RESET"
  fi
  echo
}


DOMAIN=${1:-localhost}
CERTFLD=${2:-../certs}

# =================================================
title "ÉTAPE 1 — CA (une seule fois)"
# =================================================
if [ -f ${CERTFLD}/CA.key ] && [ -f ${CERTFLD}/CA.crt ]; then
    echo "✅ CA existante - réutilisation - on vous évite de le recharger dans votre navigateur ou certif cabinet sur windows"
else
    echo "=== Génération de la CA ==="

    openssl genrsa -out ${CERTFLD}/CA.key 4096

    openssl req -new -x509 \
        -days 3650 \
        -key ${CERTFLD}/CA.key \
        -out ${CERTFLD}/CA.crt \
        -subj "/CN=MLOps-CA/O=MLOps/C=FR"

    export res=$?;result "$res" "CA générée dans ${CERTFLD}/"
    echo ""
fi

# =================================================
title "ÉTAPE 2 - importez ou distribué le CA cert (une seule fois) sur votre machine - ou celle des collègues"
# =================================================
echo "⚠️  IMPORTANT : si pas encore fait, importe CA.crt dans ton navigateur suivant l'os par tout moyen"
echo "     eg: Chrome/Linux:"
echo "     sudo cp ${CERTFLD}/CA.crt /usr/local/share/ca-certificates/"
echo "     sudo update-ca-certificates"    

# =================================================
title "ÉTAPE 3 - Fichier de config SAN obligatoire"
# =================================================
cat > ${CERTFLD}/openssl.cnf << EOF
[ req ]
distinguished_name = req_distinguished_name
req_extensions     = v3_req
prompt             = no

[ req_distinguished_name ]
CN = ${DOMAIN}
O  = MLOps
C  = FR

[ v3_req ]
subjectAltName = @alt_names
keyUsage       = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[ alt_names ]
#DNS.1 = ${DOMAIN}
DNS.2 = localhost
IP.1  = 127.0.0.1
EOF

cat ${CERTFLD}/openssl.cnf

# ============================================
title "ÉTAPE 4 - Certificat nginx signé par la CA"
# ============================================
title "=== Génération du certificat nginx ==="

openssl genrsa -out ${CERTFLD}/nginx.key 4096

openssl req -new \
    -key ${CERTFLD}/nginx.key \
    -out ${CERTFLD}/nginx.csr \
    -config ${CERTFLD}/openssl.cnf

openssl x509 -req \
    -days 365 \
    -in ${CERTFLD}/nginx.csr \
    -CA ${CERTFLD}/CA.crt \
    -CAkey ${CERTFLD}/CA.key \
    -CAcreateserial \
    -out ${CERTFLD}/nginx.crt \
    -extensions v3_req \
    -extfile ${CERTFLD}/openssl.cnf      # ← SAN injecté ici

# ============================================
title "ÉTAPE 5 - htpasswd"
# ============================================
htpasswdFILE=${CERTFLD}/../.htpasswd
if [ -f $htpasswdFILE ]; then
    echo "✅ $htpasswdFILE existant — réutilisation"
    echo "Pour d'autres accounts :"
    echo "htpasswd -c $htpasswdFILE paulbismuth"
else
    echo "=== Création mot de passe admin nginx ==="
    htpasswd -c $htpasswdFILE admin
fi
title "=== ✅ Secrets générés dans ${CERTFLD}/  ==="

# =================================================
title "ÉTAPE 6 - Nettoyage et listing"
# =================================================
rm -f ${CERTFLD}/openssl.cnf
chmod 600 ${CERTFLD}/*.key
(cd ${CERTFLD};ls -la CA.key CA.crt nginx.key nginx.csr nginx.crt .htpasswd)

# =================================================
title "ÉTAPE 7 - Exemple de commande de lecture des certificats"
# =================================================

title "=== Vérification complète ==="
title "--- KEY ---"
openssl rsa -in ${CERTFLD}/nginx.key -text -noout | grep "Private-Key"

title "--- CSR --- Vérification du SAN du certificat (Subject Alternative Name)"
openssl req -in ${CERTFLD}/nginx.csr -text -noout | grep -A 3 "Subject Alternative Name"

title "--- CRT ---"
openssl x509 -in ${CERTFLD}/nginx.crt -text -noout | grep -E "Issuer|Subject|Not After|Alternative Name|DNS|IP"

title "=== Commande openssl plus générique ==============="
echo "openssl rsa -in ${CERTFLD}/nginx.key -text -noout" 
echo "openssl req -in ${CERTFLD}nginx.csr -text -noout "
echo "openssl x509 -in ${CERTFLD}/nginx.crt -text -noout"

title "⚠️ Vérifiez TOUJOURS que .gitignore contient *.key *.csr .htpasswd pour les exclures de github"

