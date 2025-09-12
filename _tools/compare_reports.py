from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/compare_reports.py

def get_model_name_from_path(filepath):
    basename = os.path.basename(filepath)
    name = basename.replace('report_', '').replace('_on_universe.csv', '')
    return name

def plot_comparison_scatter(df1, df2, name1, name2, output_filename):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(df1['avg_recall'], df1['avg_precision'], alpha=0.6, label=name1, s=50, color='royalblue')
    median_recall1, median_precision1 = df1['avg_recall'].median(), df1['avg_precision'].median()
    ax.axvline(median_recall1, color='royalblue', linestyle='--', linewidth=1.5, label=f'Медиана Recall ({name1}): {median_recall1:.1f}%')
    ax.axhline(median_precision1, color='royalblue', linestyle=':', linewidth=1.5, label=f'Медиана Precision ({name1}): {median_precision1:.1f}%')
    ax.scatter(df2['avg_recall'], df2['avg_precision'], alpha=0.6, label=name2, s=50, color='darkorange')
    median_recall2, median_precision2 = df2['avg_recall'].median(), df2['avg_precision'].median()
    ax.axvline(median_recall2, color='darkorange', linestyle='--', linewidth=1.5, label=f'Медиана Recall ({name2}): {median_recall2:.1f}%')
    ax.axhline(median_precision2, color='darkorange', linestyle=':', linewidth=1.5, label=f'Медиана Precision ({name2}): {median_precision2:.1f}%')
    ax.set_title(f'Сравнение моделей: {name1} vs {name2}', fontsize=18, pad=20)
    ax.set_xlabel('Полнота (Recall, %)', fontsize=14)
    ax.set_ylabel('Точность (Precision, %)', fontsize=14)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.legend(loc='best', fontsize=11); ax.grid(True)
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"✅ График сравнения сохранен в файл: {output_filename}")

def compare_model_reports(file1, file2):
    try:
        df1, df2 = pd.read_csv(file1), pd.read_csv(file2)
        print(f"✅ Файлы '{os.path.basename(file1)}' и '{os.path.basename(file2)}' успешно загружены.")
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}"); return
    name1, name2 = get_model_name_from_path(file1), get_model_name_from_path(file2)
    metrics1 = {"Точность (Precision)": f"{df1['avg_precision'].median():.2f}%", "Полнота (Recall)": f"{df1['avg_recall'].median():.2f}%", "F1-Score": f"{df1['f1_score'].median():.2f}"}
    metrics2 = {"Точность (Precision)": f"{df2['avg_precision'].median():.2f}%", "Полнота (Recall)": f"{df2['avg_recall'].median():.2f}%", "F1-Score": f"{df2['f1_score'].median():.2f}"}
    summary_df = pd.DataFrame([metrics1, metrics2], index=[name1, name2])
    print("\n--- Сводная таблица сравнения моделей (медианные значения) ---"); print(summary_df)
    comparison_df = pd.merge(df1[['ticker', 'f1_score']], df2[['ticker', 'f1_score']], on='ticker', suffixes=(f'_{name1}', f'_{name2}'))
    comparison_df['f1_improvement'] = comparison_df[f'f1_score_{name2}'] - comparison_df[f'f1_score_{name1}']
    improved_count = (comparison_df['f1_improvement'] > 0).sum()
    declined_count = (comparison_df['f1_improvement'] < 0).sum()
    unchanged_count = len(comparison_df) - improved_count - declined_count
    print("\n--- Детальный анализ по акциям (изменение F1-Score) ---")
    print(f"Улучшение F1-Score: {improved_count} акций"); print(f"Ухудшение F1-Score: {declined_count} акций"); print(f"Без изменений: {unchanged_count} акций")
    results_dir = os.path.dirname(file1)
    plot_filename = os.path.join(results_dir, f"comparison_{name1}_vs_{name2}.png")
    plot_comparison_scatter(df1, df2, name1, name2, plot_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для сравнения двух отчетов о производительности моделей.")
    parser.add_argument('report1', type=str, help='Путь к первому файлу отчета (CSV).')
    parser.add_argument('report2', type=str, help='Путь ко второму файлу отчета (CSV).')
    args = parser.parse_args()
    compare_model_reports(args.report1, args.report2)
