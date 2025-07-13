"""
Simplified Comprehensive Data Analysis Service
Using basic pandas and statistics for demonstration
"""

import pandas as pd
import numpy as np
import io
import base64
import tempfile
import os
from pathlib import Path
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveDataAnalyzer:
    """
    Simplified comprehensive data analysis demonstrating the concept
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "ai_data_analysis"
        self.temp_dir.mkdir(exist_ok=True)
    
    def analyze_dataset(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Run comprehensive analysis using basic pandas analytics
        
        Args:
            df: The pandas DataFrame to analyze
            filename: Original filename for context
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Generate unique session ID for this analysis
            session_id = f"analysis_{hash(filename)}_{len(df)}"
            
            # Run all three analyses (simplified versions)
            profiling_result = self._run_data_profiling(df, filename, session_id)
            validation_result = self._run_data_validation(df, filename, session_id)
            exploration_result = self._run_data_exploration(df, filename, session_id)
            
            # Combine results with summary
            combined_result = {
                "session_id": session_id,
                "filename": filename,
                "dataset_info": {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "memory_usage": f"{df.memory_usage().sum() / 1024 / 1024:.2f} MB"
                },
                "data_profiling": profiling_result,
                "data_validation": validation_result,
                "data_exploration": exploration_result,
                "executive_summary": self._generate_executive_summary(df, profiling_result, validation_result, exploration_result)
            }
            
            return combined_result
            
        except Exception as e:
            return {
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None,
                "partial_results": True
            }
    
    def _run_data_profiling(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Run basic data profiling - Complete data understanding
        """
        try:
            # Basic profiling statistics
            profiling_stats = {
                "data_types": {},
                "missing_values": {},
                "basic_stats": {},
                "unique_values": {},
                "correlation_analysis": {}
            }
            
            # Data types analysis
            for col in df.columns:
                profiling_stats["data_types"][col] = {
                    "dtype": str(df[col].dtype),
                    "type_category": self._categorize_column_type(df[col])
                }
            
            # Missing values analysis
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                profiling_stats["missing_values"][col] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_count / len(df) * 100)
                }
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                profiling_stats["basic_stats"] = df[numeric_cols].describe().to_dict()
                
                # Correlation analysis
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    profiling_stats["correlation_analysis"] = corr_matrix.to_dict()
            
            # Unique values analysis
            for col in df.columns:
                profiling_stats["unique_values"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "unique_percentage": float(df[col].nunique() / len(df) * 100)
                }
            
            result = {
                "analysis_type": "data-profiling",
                "title": "Complete Data Understanding",
                "status": "success",
                "key_insights": {
                    "total_columns": len(df.columns),
                    "total_rows": len(df),
                    "numeric_columns": len(numeric_cols),
                    "text_columns": len(df.select_dtypes(include=['object']).columns),
                    "missing_data_columns": len([col for col in df.columns if df[col].isnull().any()]),
                    "duplicate_rows": int(df.duplicated().sum())
                },
                "detailed_stats": profiling_stats,
                "recommendations": self._generate_profiling_recommendations(df, profiling_stats)
            }
            
            return result
            
        except Exception as e:
            return {
                "analysis_type": "data-profiling",
                "status": "error",
                "error": str(e),
                "message": "Failed to generate data profiling report"
            }
    
    def _run_data_validation(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Run basic data validation - Data quality checks
        """
        try:
            validation_results = []
            
            # Check 1: Dataset not empty
            validation_results.append({
                "check": "Dataset not empty",
                "status": "pass" if len(df) > 0 else "fail",
                "details": f"Dataset contains {len(df)} rows"
            })
            
            # Check 2: No completely empty columns
            empty_cols = [col for col in df.columns if df[col].isnull().all()]
            validation_results.append({
                "check": "No completely empty columns",
                "status": "pass" if len(empty_cols) == 0 else "fail",
                "details": f"Found {len(empty_cols)} completely empty columns" if empty_cols else "All columns contain data"
            })
            
            # Check 3: Reasonable missing value percentages
            high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.5]
            validation_results.append({
                "check": "Acceptable missing value levels",
                "status": "pass" if len(high_missing_cols) == 0 else "warning",
                "details": f"Found {len(high_missing_cols)} columns with >50% missing values" if high_missing_cols else "Missing value levels acceptable"
            })
            
            # Check 4: Data type consistency
            mixed_type_cols = []
            for col in df.select_dtypes(include=['object']).columns:
                # Simple check for mixed types in object columns
                try:
                    pd.to_numeric(df[col].dropna())
                    mixed_type_cols.append(col)
                except:
                    pass
            
            validation_results.append({
                "check": "Data type consistency",
                "status": "warning" if len(mixed_type_cols) > 0 else "pass",
                "details": f"Found {len(mixed_type_cols)} columns with potential type inconsistencies" if mixed_type_cols else "Data types appear consistent"
            })
            
            # Check 5: Duplicate detection
            duplicate_count = df.duplicated().sum()
            validation_results.append({
                "check": "Duplicate row detection",
                "status": "warning" if duplicate_count > 0 else "pass",
                "details": f"Found {duplicate_count} duplicate rows" if duplicate_count > 0 else "No duplicate rows detected"
            })
            
            # Calculate overall quality score
            pass_count = len([r for r in validation_results if r["status"] == "pass"])
            warning_count = len([r for r in validation_results if r["status"] == "warning"])
            fail_count = len([r for r in validation_results if r["status"] == "fail"])
            
            total_checks = len(validation_results)
            quality_score = ((pass_count * 1.0 + warning_count * 0.5 + fail_count * 0.0) / total_checks * 100)
            
            result = {
                "analysis_type": "data-validation",
                "title": "Data Quality Validation",
                "status": "success",
                "overall_quality_score": quality_score,
                "validation_results": validation_results,
                "summary": {
                    "total_checks": total_checks,
                    "passed_checks": pass_count,
                    "warning_checks": warning_count,
                    "failed_checks": fail_count,
                    "quality_grade": self._get_quality_grade(quality_score)
                },
                "recommendations": self._generate_validation_recommendations(validation_results, quality_score)
            }
            
            return result
            
        except Exception as e:
            return {
                "analysis_type": "data-validation",
                "status": "error",
                "error": str(e),
                "message": "Failed to run data validation"
            }
    
    def _run_data_exploration(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Run basic data exploration - Visual insights
        """
        try:
            exploration_insights = {
                "feature_analysis": {},
                "distribution_analysis": {},
                "relationship_analysis": {},
                "outlier_analysis": {}
            }
            
            # Feature analysis
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            exploration_insights["feature_analysis"] = {
                "total_features": len(df.columns),
                "categorical_features": len(categorical_cols),
                "numerical_features": len(numerical_cols),
                "datetime_features": len(datetime_cols),
                "feature_breakdown": {
                    "categorical": list(categorical_cols),
                    "numerical": list(numerical_cols),
                    "datetime": list(datetime_cols)
                }
            }
            
            # Distribution analysis for categorical columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                exploration_insights["distribution_analysis"][col] = {
                    "type": "categorical",
                    "unique_values": int(df[col].nunique()),
                    "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "cardinality": "high" if df[col].nunique() > len(df) * 0.5 else "low"
                }
            
            # Distribution analysis for numerical columns
            for col in numerical_cols:
                exploration_insights["distribution_analysis"][col] = {
                    "type": "numerical",
                    "mean": float(df[col].mean()) if not df[col].empty else None,
                    "median": float(df[col].median()) if not df[col].empty else None,
                    "std": float(df[col].std()) if not df[col].empty else None,
                    "skewness": float(df[col].skew()) if not df[col].empty else None,
                    "kurtosis": float(df[col].kurtosis()) if not df[col].empty else None
                }
            
            # Relationship analysis (correlations)
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                strong_correlations = []
                
                for i, col1 in enumerate(numerical_cols):
                    for j, col2 in enumerate(numerical_cols):
                        if i < j:  # Avoid duplicates
                            corr_value = corr_matrix.loc[col1, col2]
                            if abs(corr_value) > 0.7:  # Strong correlation threshold
                                strong_correlations.append({
                                    "variable1": col1,
                                    "variable2": col2,
                                    "correlation": float(corr_value),
                                    "strength": "strong" if abs(corr_value) > 0.8 else "moderate"
                                })
                
                exploration_insights["relationship_analysis"] = {
                    "correlation_matrix_available": True,
                    "strong_correlations": strong_correlations
                }
            
            # Outlier analysis
            outlier_summary = {}
            for col in numerical_cols:
                outliers = self._detect_outliers_iqr(df[col])
                outlier_summary[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(df) * 100)
                }
            
            exploration_insights["outlier_analysis"] = outlier_summary
            
            result = {
                "analysis_type": "data-exploration",
                "title": "Visual Data Exploration",
                "status": "success",
                "insights": exploration_insights,
                "summary": {
                    "data_completeness": float((df.notna().sum().sum()) / (len(df) * len(df.columns)) * 100),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "potential_outliers": sum([info["outlier_count"] for info in outlier_summary.values()])
                },
                "recommendations": self._generate_exploration_recommendations(exploration_insights)
            }
            
            return result
            
        except Exception as e:
            return {
                "analysis_type": "data-exploration",
                "status": "error",
                "error": str(e),
                "message": "Failed to generate data exploration report"
            }
    
    def _categorize_column_type(self, series: pd.Series) -> str:
        """Categorize column type"""
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_categorical_dtype(series):
            return "categorical"
        else:
            return "text"
    
    def _detect_outliers_iqr(self, series: pd.Series) -> list:
        """Detect outliers using IQR method"""
        try:
            if not pd.api.types.is_numeric_dtype(series):
                return []
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return outliers.index.tolist()
        except:
            return []
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Fair)"
        elif score >= 60:
            return "D (Poor)"
        else:
            return "F (Critical)"
    
    def _generate_profiling_recommendations(self, df: pd.DataFrame, stats: Dict) -> list:
        """Generate recommendations based on profiling results"""
        recommendations = []
        
        # Check for high missing values
        missing_values = stats.get("missing_values", {})
        for col, info in missing_values.items():
            missing_pct = info.get("missing_percentage", 0)
            if missing_pct > 50:
                recommendations.append(f"⚠️ Column '{col}' has {missing_pct:.1f}% missing values - consider imputation or removal")
            elif missing_pct > 20:
                recommendations.append(f"📊 Column '{col}' has {missing_pct:.1f}% missing values - review data collection process")
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            dup_pct = df.duplicated().sum() / len(df) * 100
            recommendations.append(f"🔍 Dataset contains {dup_pct:.1f}% duplicate rows - consider deduplication")
        
        # Check for high cardinality categorical columns
        unique_values = stats.get("unique_values", {})
        for col, info in unique_values.items():
            if df[col].dtype == 'object' and info.get("unique_percentage", 0) > 90:
                recommendations.append(f"📈 Column '{col}' has very high cardinality - consider grouping strategies")
        
        if not recommendations:
            recommendations.append("✅ Data profiling looks good - no major issues detected")
        
        return recommendations
    
    def _generate_validation_recommendations(self, results: list, quality_score: float) -> list:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if quality_score < 70:
            recommendations.append("🚨 Data quality score is below acceptable threshold - immediate attention required")
        elif quality_score < 85:
            recommendations.append("⚠️ Data quality needs improvement - review failed validations")
        else:
            recommendations.append("✅ Good data quality detected - suitable for analysis")
        
        # Analyze failed/warning results
        issues = [r for r in results if r["status"] in ["fail", "warning"]]
        if issues:
            recommendations.append(f"🔧 {len(issues)} validation issue(s) detected - review data preprocessing")
        
        return recommendations
    
    def _generate_exploration_recommendations(self, insights: Dict) -> list:
        """Generate recommendations based on exploration insights"""
        recommendations = []
        
        feature_analysis = insights.get("feature_analysis", {})
        outlier_analysis = insights.get("outlier_analysis", {})
        
        # Check feature balance
        categorical_features = feature_analysis.get("categorical_features", 0)
        numerical_features = feature_analysis.get("numerical_features", 0)
        
        if categorical_features == 0:
            recommendations.append("📊 Dataset contains only numerical features - consider feature engineering")
        elif numerical_features == 0:
            recommendations.append("📈 Dataset contains only categorical features - consider encoding strategies")
        
        # Check for outliers
        total_outliers = sum([info.get("outlier_count", 0) for info in outlier_analysis.values()])
        if total_outliers > 0:
            recommendations.append(f"📉 Detected {total_outliers} potential outliers across numerical columns")
        
        # Check relationships
        relationship_analysis = insights.get("relationship_analysis", {})
        strong_correlations = relationship_analysis.get("strong_correlations", [])
        if len(strong_correlations) > 0:
            recommendations.append(f"🔗 Found {len(strong_correlations)} strong correlations - review for multicollinearity")
        
        if not recommendations:
            recommendations.append("📋 Data exploration complete - dataset ready for analysis")
        
        return recommendations
    
    def _generate_executive_summary(self, df: pd.DataFrame, profiling_result: Dict, 
                                  validation_result: Dict, exploration_result: Dict) -> Dict[str, Any]:
        """Generate an executive summary combining all three analyses"""
        
        # Overall data quality assessment
        quality_scores = []
        
        if validation_result.get('status') == 'success':
            quality_scores.append(validation_result.get('overall_quality_score', 0))
        
        if exploration_result.get('status') == 'success':
            completeness = exploration_result.get('summary', {}).get('data_completeness', 0)
            quality_scores.append(completeness)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Key findings
        key_findings = []
        
        # From profiling
        if profiling_result.get('status') == 'success':
            insights = profiling_result.get('key_insights', {})
            if insights.get('missing_data_columns', 0) > 0:
                key_findings.append(f"Missing values detected in {insights['missing_data_columns']} columns")
            if insights.get('duplicate_rows', 0) > 0:
                key_findings.append(f"{insights['duplicate_rows']} duplicate rows found")
        
        # From validation
        if validation_result.get('status') == 'success':
            failed_checks = validation_result.get('summary', {}).get('failed_checks', 0)
            if failed_checks > 0:
                key_findings.append(f"{failed_checks} data quality validations failed")
        
        # From exploration
        if exploration_result.get('status') == 'success':
            outliers = exploration_result.get('summary', {}).get('potential_outliers', 0)
            if outliers > 0:
                key_findings.append(f"{outliers} potential outliers detected")
        
        if not key_findings:
            key_findings.append("No major data quality issues detected")
        
        # Priority recommendations
        priority_recommendations = []
        
        for result in [profiling_result, validation_result, exploration_result]:
            if result.get('status') == 'success':
                recs = result.get('recommendations', [])
                # Get high priority recommendations (those with warning symbols)
                priority_recs = [rec for rec in recs if any(symbol in rec for symbol in ['🚨', '⚠️', '🔧'])]
                priority_recommendations.extend(priority_recs[:2])  # Limit to 2 per analysis
        
        summary = {
            "overall_quality_score": avg_quality,
            "quality_grade": self._get_quality_grade(avg_quality),
            "dataset_size": f"{df.shape[0]:,} rows × {df.shape[1]} columns",
            "analysis_completion": {
                "data_profiling": profiling_result.get('status') == 'success',
                "data_validation": validation_result.get('status') == 'success',
                "data_exploration": exploration_result.get('status') == 'success'
            },
            "key_findings": key_findings[:5],  # Limit to 5 key findings
            "priority_recommendations": priority_recommendations[:5],  # Limit to 5 priority recommendations
            "next_steps": [
                "Review the detailed analysis reports",
                "Address high-priority data quality issues",
                "Implement recommended data preprocessing steps",
                "Begin statistical analysis with clean data"
            ]
        }
        
        return summary