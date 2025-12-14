#!/bin/bash

# Скрипт для создания сводки всего проекта в один текстовый файл
# Использование: ./project2txt.sh [путь_к_проекту] [выходной_файл]

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Параметры по умолчанию
PROJECT_PATH="${1:-.}"  # Текущая директория по умолчанию
OUTPUT_FILE="${2:-project.txt}"  # project.txt по умолчанию
MAX_FILE_SIZE_MB=10  # Максимальный размер файла в МБ

# Расширения файлов для включения
FILE_EXTENSIONS=("py" "js" "html" "css" "json" "md" "txt" "yml" "yaml" "toml" "sh" "conf" "ini" "cfg")

# Директории для исключения
EXCLUDE_DIRS=("node_modules" "__pycache__" ".git" "venv" "env" ".pytest_cache" "dist" "build" ".env" "target" ".idea" ".vscode")

# Статистика
total_files=0
included_files=0
total_size=0

# Функция для проверки, должна ли директория быть исключена
should_exclude_dir() {
    local dir="$1"
    for exclude_dir in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$dir" == *"$exclude_dir"* ]]; then
            return 0
        fi
    done
    return 1
}

# Функция для проверки расширения файла
has_allowed_extension() {
    local file="$1"
    local ext="${file##*.}"
    for allowed_ext in "${FILE_EXTENSIONS[@]}"; do
        if [[ "$ext" == "$allowed_ext" ]]; then
            return 0
        fi
    done
    return 1
}

# Функция для форматирования размера
format_size() {
    local size=$1
    if [[ $size -lt 1024 ]]; then
        echo "${size}B"
    elif [[ $size -lt 1048576 ]]; then
        echo "$((size / 1024))KB"
    else
        echo "$(echo "scale=2; $size / 1048576" | bc)MB"
    fi
}

echo -e "${BLUE}🔍 Сканирование проекта: $PROJECT_PATH${NC}"
echo -e "${BLUE}📄 Выходной файл: $OUTPUT_FILE${NC}"
echo -e "${BLUE}📊 Фильтр по расширениям: $(IFS=,; echo "${FILE_EXTENSIONS[*]}")${NC}"
echo -e "${BLUE}🚫 Исключенные директории: $(IFS=,; echo "${EXCLUDE_DIRS[*]}")${NC}"
echo -e "${BLUE}💾 Макс. размер файла: ${MAX_FILE_SIZE_MB}МБ${NC}"
echo "============================================================================"

# Очищаем выходной файл
echo "" > "$OUTPUT_FILE"

# Записываем заголовок
cat >> "$OUTPUT_FILE" << EOF
================================================================================
ПРОЕКТ: $(basename "$PROJECT_PATH")
Дата создания: $(date '+%Y-%m-%d %H:%M:%S')
Путь к проекту: $(realpath "$PROJECT_PATH")
================================================================================

EOF

# Используем find для поиска всех файлов
while IFS= read -r -d '' file; do
    # Получаем информацию о файле
    file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    file_dir=$(dirname "$file")
    file_name=$(basename "$file")
    
    # Проверяем, не является ли файл выходным
    if [[ "$file" == *"$OUTPUT_FILE" ]]; then
        continue
    fi
    
    # Проверяем директорию на исключение
    if should_exclude_dir "$file_dir"; then
        continue
    fi
    
    # Проверяем расширение
    if ! has_allowed_extension "$file"; then
        continue
    fi
    
    # Проверяем размер файла
    max_size_bytes=$((MAX_FILE_SIZE_MB * 1024 * 1024))
    if [[ $file_size -gt $max_size_bytes ]]; then
        echo -e "${YELLOW}⚠️  Пропущен (слишком большой): $(realpath --relative-to="$PROJECT_PATH" "$file") ($(format_size $file_size))${NC}"
        continue
    fi
    
    # Пытаемся прочитать файл как текст
    if file "$file" | grep -q "text"; then
        relative_path=$(realpath --relative-to="$PROJECT_PATH" "$file" 2>/dev/null || echo "$file")
        
        # Записываем информацию о файле
        cat >> "$OUTPUT_FILE" << EOF
================================================================================
📁 ФАЙЛ: $relative_path
📊 Размер: $file_size байт ($(format_size $file_size))
📂 Полный путь: $(realpath "$file")
================================================================================

EOF
        
        # Записываем содержимое файла
        cat "$file" >> "$OUTPUT_FILE"
        echo -e "\n\n" >> "$OUTPUT_FILE"
        
        ((total_files++))
        ((included_files++))
        total_size=$((total_size + file_size))
        
        echo -e "${GREEN}✅ Добавлен: $relative_path${NC}"
    else
        echo -e "${YELLOW}⚠️  Пропущен (не текстовый): $(realpath --relative-to="$PROJECT_PATH" "$file")${NC}"
    fi
done < <(find "$PROJECT_PATH" -type f -print0 2>/dev/null)

# Записываем статистику
cat >> "$OUTPUT_FILE" << EOF
================================================================================
📊 СТАТИСТИКА ПРОЕКТА
================================================================================
Всего файлов в проекте: $total_files
Включено файлов: $included_files
Общий размер включенных файлов: $(format_size $total_size)
Фильтр по расширениям: $(IFS=,; echo "${FILE_EXTENSIONS[*]}")
Исключенные директории: $(IFS=,; echo "${EXCLUDE_DIRS[*]}")
================================================================================
EOF

echo "============================================================================"
echo -e "${GREEN}✅ Готово! Файл создан: $OUTPUT_FILE${NC}"
echo -e "📊 Всего файлов: $total_files"
echo -e "📄 Включено файлов: $included_files"
echo -e "💾 Общий размер: $(format_size $total_size)"
echo -e "📁 Результат сохранен в: $(realpath "$OUTPUT_FILE")"

