"""
Incoming: evaluation results --- {Dict, DataFrame}
Processing: graph generation --- {matplotlib, seaborn}
Outgoing: publication-quality figures --- {PNG, PDF, SVG}

Research Paper Visualization Utility
------------------------------------
Publication-quality figures for IR/NLP research papers.
Follows ACL/EMNLP/SIGIR formatting conventions.

STRICT: No inline magic numbers. All styling via configuration.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Try seaborn, graceful fallback
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# Configuration: Research Paper Styling
# =============================================================================

class FigureStyle:
    """Publication-ready figure styling configuration."""
    
    # ACL/EMNLP column widths (inches)
    SINGLE_COLUMN = 3.25
    DOUBLE_COLUMN = 6.75
    
    # Font sizes (LaTeX compatible)
    FONT_SIZE_SMALL = 8
    FONT_SIZE_MEDIUM = 9
    FONT_SIZE_LARGE = 10
    FONT_SIZE_TITLE = 11
    
    # Default figure sizes (width, height) in inches
    FIGURE_SIZES = {
        'single': (SINGLE_COLUMN, 2.5),
        'single_tall': (SINGLE_COLUMN, 3.5),
        'double': (DOUBLE_COLUMN, 3.0),
        'double_wide': (DOUBLE_COLUMN, 4.0),
        'square': (SINGLE_COLUMN, SINGLE_COLUMN),
    }
    
    # Color palettes (colorblind-friendly)
    PALETTES = {
        'default': ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD'],
        'colorblind': ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9'],
        'grayscale': ['#000000', '#404040', '#808080', '#B0B0B0', '#D0D0D0'],
        'academic': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    }
    
    # Marker styles
    MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    # Line styles
    LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Hatch patterns for bar charts
    HATCHES = ['', '/', '\\', 'x', '-', '+', '.', 'o']


def setup_matplotlib_style(use_latex: bool = False):
    """
    Configure matplotlib for publication-quality figures.
    
    Args:
        use_latex: Use LaTeX rendering (requires TeX installation)
    """
    plt.rcParams.update({
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Font
        'font.family': 'serif' if use_latex else 'sans-serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': FigureStyle.FONT_SIZE_MEDIUM,
        
        # Axes
        'axes.titlesize': FigureStyle.FONT_SIZE_LARGE,
        'axes.labelsize': FigureStyle.FONT_SIZE_MEDIUM,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        
        # Ticks
        'xtick.labelsize': FigureStyle.FONT_SIZE_SMALL,
        'ytick.labelsize': FigureStyle.FONT_SIZE_SMALL,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend
        'legend.fontsize': FigureStyle.FONT_SIZE_SMALL,
        'legend.frameon': False,
        'legend.loc': 'best',
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        
        # Grid
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # LaTeX
        'text.usetex': use_latex,
        'text.latex.preamble': r'\usepackage{amsmath}' if use_latex else '',
    })
    
    if HAS_SEABORN:
        sns.set_style('whitegrid', {
            'axes.edgecolor': '0.2',
            'axes.facecolor': 'white',
            'grid.color': '0.9',
            'grid.linestyle': '-',
        })


# =============================================================================
# Core Plotting Functions
# =============================================================================

class ResearchFigure:
    """
    Context manager for creating publication-ready figures.
    
    Usage:
        with ResearchFigure(size='single', output_path='fig1.pdf') as fig:
            ax = fig.add_subplot(111)
            ax.plot(x, y)
    """
    
    def __init__(
        self,
        size: str = 'single',
        figsize: Optional[Tuple[float, float]] = None,
        output_path: Optional[str] = None,
        formats: List[str] = ['pdf', 'png'],
        use_latex: bool = False
    ):
        self.size = size
        self.figsize = figsize or FigureStyle.FIGURE_SIZES.get(size, (3.25, 2.5))
        self.output_path = output_path
        self.formats = formats
        self.use_latex = use_latex
        self.fig = None
    
    def __enter__(self):
        setup_matplotlib_style(self.use_latex)
        self.fig = plt.figure(figsize=self.figsize)
        return self.fig
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.output_path:
            self.save()
        plt.close(self.fig)
    
    def save(self):
        """Save figure in multiple formats."""
        base_path = Path(self.output_path).with_suffix('')
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in self.formats:
            path = base_path.with_suffix(f'.{fmt}')
            self.fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
            print(f"[visualization] Saved: {path}")


def bar_comparison(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    output_path: Optional[str] = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    y_lim: Optional[Tuple[float, float]] = None,
    legend_labels: Optional[List[str]] = None,
    palette: str = 'default',
    annotate: bool = True,
    figsize: str = 'single_tall',
    horizontal: bool = False,
    sort_by: Optional[str] = None,
    highlight_best: bool = True
) -> plt.Figure:
    """
    Create grouped bar chart for method comparison.
    
    Args:
        data: DataFrame with results
        x_col: Column for x-axis categories
        y_cols: Columns to plot as bars
        output_path: Save path (optional)
        title: Figure title
        xlabel, ylabel: Axis labels
        y_lim: Y-axis limits
        legend_labels: Custom legend labels
        palette: Color palette name
        annotate: Show values on bars
        figsize: Figure size preset
        horizontal: Horizontal bars
        sort_by: Sort by this column (descending)
        highlight_best: Bold/mark best value
        
    Returns:
        matplotlib Figure
    """
    setup_matplotlib_style()
    
    if sort_by and sort_by in data.columns:
        data = data.sort_values(sort_by, ascending=False)
    
    colors = FigureStyle.PALETTES.get(palette, FigureStyle.PALETTES['default'])
    fig_size = FigureStyle.FIGURE_SIZES.get(figsize, (3.25, 3.5))
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    x = np.arange(len(data[x_col]))
    n_bars = len(y_cols)
    width = 0.8 / n_bars
    
    for i, col in enumerate(y_cols):
        offset = (i - n_bars / 2 + 0.5) * width
        values = data[col].values
        
        if horizontal:
            bars = ax.barh(x + offset, values, width, label=legend_labels[i] if legend_labels else col,
                          color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
        else:
            bars = ax.bar(x + offset, values, width, label=legend_labels[i] if legend_labels else col,
                         color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
        
        if annotate:
            for bar, val in zip(bars, values):
                if horizontal:
                    ax.annotate(f'{val:.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                               xytext=(2, 0), textcoords='offset points', va='center', fontsize=6)
                else:
                    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=6)
    
    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(data[x_col])
        ax.set_xlabel(ylabel or 'Score')
        ax.set_ylabel(xlabel)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(data[x_col], rotation=45, ha='right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or 'Score')
    
    if y_lim:
        if horizontal:
            ax.set_xlim(y_lim)
        else:
            ax.set_ylim(y_lim)
    
    if title:
        ax.set_title(title, fontsize=FigureStyle.FONT_SIZE_TITLE, fontweight='bold')
    
    if n_bars > 1:
        ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def line_plot(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    output_path: Optional[str] = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    y_lim: Optional[Tuple[float, float]] = None,
    legend_labels: Optional[List[str]] = None,
    palette: str = 'default',
    markers: bool = True,
    figsize: str = 'single',
    show_error: bool = False,
    error_cols: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create line plot for trends (e.g., k-shot performance).
    
    Args:
        data: DataFrame with x values and y series
        x_col: Column for x-axis
        y_cols: Columns to plot as lines
        output_path: Save path (optional)
        title: Figure title
        xlabel, ylabel: Axis labels
        y_lim: Y-axis limits
        legend_labels: Custom legend labels
        palette: Color palette name
        markers: Show data point markers
        figsize: Figure size preset
        show_error: Show error bars/bands
        error_cols: Error value columns (same order as y_cols)
        
    Returns:
        matplotlib Figure
    """
    setup_matplotlib_style()
    
    colors = FigureStyle.PALETTES.get(palette, FigureStyle.PALETTES['default'])
    fig_size = FigureStyle.FIGURE_SIZES.get(figsize, (3.25, 2.5))
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    x = data[x_col].values
    
    for i, col in enumerate(y_cols):
        y = data[col].values
        label = legend_labels[i] if legend_labels else col
        color = colors[i % len(colors)]
        marker = FigureStyle.MARKERS[i % len(FigureStyle.MARKERS)] if markers else None
        linestyle = FigureStyle.LINE_STYLES[i % len(FigureStyle.LINE_STYLES)]
        
        ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle)
        
        if show_error and error_cols and i < len(error_cols):
            err = data[error_cols[i]].values
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if y_lim:
        ax.set_ylim(y_lim)
    
    if title:
        ax.set_title(title, fontsize=FigureStyle.FONT_SIZE_TITLE, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def table_comparison(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    caption: str = '',
    label: str = '',
    highlight_best: bool = True,
    highlight_cols: Optional[List[str]] = None,
    precision: int = 4,
    format_type: str = 'latex'
) -> str:
    """
    Generate publication-ready table (LaTeX or Markdown).
    
    Args:
        data: DataFrame with results
        output_path: Save path for .tex or .md file
        caption: Table caption
        label: LaTeX label for referencing
        highlight_best: Bold best value per column
        highlight_cols: Columns to apply highlighting
        precision: Decimal precision
        format_type: 'latex' or 'markdown'
        
    Returns:
        Table string
    """
    df = data.copy()
    
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    highlight_cols = highlight_cols or numeric_cols
    
    if highlight_best:
        for col in highlight_cols:
            if col in df.columns:
                max_val = df[col].max()
                df[col] = df[col].apply(
                    lambda x: f'\\textbf{{{x:.{precision}f}}}' if x == max_val else f'{x:.{precision}f}'
                    if format_type == 'latex'
                    else f'**{x:.{precision}f}**' if x == max_val else f'{x:.{precision}f}'
                )
    else:
        for col in numeric_cols:
            df[col] = df[col].apply(lambda x: f'{x:.{precision}f}')
    
    if format_type == 'latex':
        # Generate LaTeX table
        col_format = 'l' + 'c' * (len(df.columns) - 1)
        
        lines = [
            '\\begin{table}[t]',
            '\\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}' if label else '',
            f'\\begin{{tabular}}{{{col_format}}}',
            '\\toprule',
        ]
        
        # Header
        header = ' & '.join(df.columns.tolist())
        lines.append(f'{header} \\\\')
        lines.append('\\midrule')
        
        # Data rows
        for _, row in df.iterrows():
            row_str = ' & '.join(str(v) for v in row.values)
            lines.append(f'{row_str} \\\\')
        
        lines.extend([
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}'
        ])
        
        table_str = '\n'.join(lines)
    
    else:  # markdown
        table_str = df.to_markdown(index=False)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
        print(f"[visualization] Saved table: {output_path}")
    
    return table_str


def heatmap(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    cmap: str = 'RdYlGn',
    annotate: bool = True,
    figsize: str = 'square',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Create heatmap for correlation/confusion matrices.
    
    Args:
        data: DataFrame (matrix form)
        output_path: Save path
        title: Figure title
        xlabel, ylabel: Axis labels
        cmap: Colormap name
        annotate: Show values in cells
        figsize: Figure size preset
        vmin, vmax: Color scale limits
        
    Returns:
        matplotlib Figure
    """
    setup_matplotlib_style()
    
    fig_size = FigureStyle.FIGURE_SIZES.get(figsize, (3.25, 3.25))
    fig, ax = plt.subplots(figsize=fig_size)
    
    if HAS_SEABORN:
        sns.heatmap(data, annot=annotate, fmt='.3f', cmap=cmap, ax=ax,
                   vmin=vmin, vmax=vmax, linewidths=0.5,
                   annot_kws={'size': 7}, cbar_kws={'shrink': 0.8})
    else:
        im = ax.imshow(data.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns, rotation=45, ha='right')
        ax.set_yticklabels(data.index)
        
        if annotate:
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    ax.annotate(f'{data.iloc[i, j]:.3f}',
                               xy=(j, i), ha='center', va='center', fontsize=7)
    
    if title:
        ax.set_title(title, fontsize=FigureStyle.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def beir_comparison_plot(
    our_results: Dict[str, float],
    beir_baselines: Dict[str, float],
    output_path: Optional[str] = None,
    title: str = 'nDCG@10 Comparison with BEIR Benchmark',
    figsize: str = 'double'
) -> plt.Figure:
    """
    Create comparison plot between our results and BEIR benchmark.
    
    Args:
        our_results: {ranker_name: ndcg@10_score}
        beir_baselines: {ranker_name: ndcg@10_score} from BEIR paper
        output_path: Save path
        title: Figure title
        figsize: Figure size preset
        
    Returns:
        matplotlib Figure
    """
    setup_matplotlib_style()
    
    fig_size = FigureStyle.FIGURE_SIZES.get(figsize, (6.75, 3.0))
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    colors = FigureStyle.PALETTES['academic']
    
    # Left: Our results
    ax1 = axes[0]
    methods = list(our_results.keys())
    values = list(our_results.values())
    
    bars = ax1.barh(methods, values, color=colors[0], edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('nDCG@10')
    ax1.set_title('Our Results', fontweight='bold')
    ax1.set_xlim(0, 0.7)
    
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=7)
    
    # Right: BEIR baselines
    ax2 = axes[1]
    beir_methods = list(beir_baselines.keys())
    beir_values = list(beir_baselines.values())
    
    bars = ax2.barh(beir_methods, beir_values, color=colors[1], edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('nDCG@10')
    ax2.set_title('BEIR Benchmark (Table 2)', fontweight='bold')
    ax2.set_xlim(0, 0.7)
    
    for bar, val in zip(bars, beir_values):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=7)
    
    fig.suptitle(title, fontsize=FigureStyle.FONT_SIZE_TITLE, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def save_figure(fig: plt.Figure, path: str, formats: List[str] = ['pdf', 'png']):
    """Save figure in multiple formats."""
    base = Path(path).with_suffix('')
    base.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fpath = base.with_suffix(f'.{fmt}')
        fig.savefig(fpath, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[visualization] Saved: {fpath}")


def create_results_df(results: Dict[str, Dict[str, float]], index_name: str = 'Method') -> pd.DataFrame:
    """
    Convert nested results dict to DataFrame.
    
    Args:
        results: {method_name: {metric: value}}
        index_name: Name for index column
        
    Returns:
        DataFrame with methods as rows, metrics as columns
    """
    df = pd.DataFrame(results).T
    df.index.name = index_name
    return df.reset_index()


def format_improvement(baseline: float, improved: float, precision: int = 2) -> str:
    """Format improvement as percentage string."""
    if baseline == 0:
        return "N/A"
    pct = (improved - baseline) / baseline * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.{precision}f}%"


