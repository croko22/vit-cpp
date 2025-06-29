GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 

usage() {
    echo -e "${YELLOW}Uso: $0 [comando] [argumento]${NC}"
    echo "Comandos:"
    echo "  build         Compila todos los ejemplos."
    echo "  run [ejemplo] Compila y ejecuta un ejemplo específico."
    echo "                Ejemplos: layernorm, multihead, feedforward, encoder, decoder, transformer"
    echo "  test          Compila y ejecuta todos los ejemplos en secuencia."
    echo "  clean         Limpia todos los archivos compilados."
    echo "  help          Muestra este mensaje de ayuda."
}

main() {
    COMMAND=$1
    ARG=$2

    case "$COMMAND" in
        build)
            echo -e "${GREEN}--- Construyendo todos los ejemplos ---${NC}"
            make all
            ;;
        run)
            if [ -z "$ARG" ]; then
                echo -e "${RED}Error: El comando 'run' requiere el nombre de un ejemplo.${NC}"
                usage
                exit 1
            fi
            echo -e "${GREEN}--- Compilando y ejecutando '$ARG' ---${NC}"
            make "$ARG" && ./build/"$ARG".out
            ;;
        test)
            echo -e "${GREEN}--- Ejecutando todas las pruebas ---${NC}"
            make all
            if [ $? -eq 0 ]; then
                for example in layernorm multihead feedforward encoder decoder transformer; do
                    echo -e "\n${YELLOW}--- Probando: $example ---${NC}"
                    ./build/"$example".out
                done
                echo -e "\n${GREEN}--- Todas las pruebas se ejecutaron con éxito ---${NC}"
            else
                echo -e "${RED}La compilación falló. No se pueden ejecutar las pruebas.${NC}"
            fi
            ;;
        clean)
            echo -e "${GREEN}--- Limpiando el proyecto ---${NC}"
            make clean
            ;;
        help|*)
            usage
            ;;
    esac
}

main "$@"