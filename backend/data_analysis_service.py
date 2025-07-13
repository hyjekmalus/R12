"""
Comprehensive Data Analysis Service
Automatically runs ydata-profiling, Great Expectations, and Sweetviz analysis
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

# Import the three analysis libraries
from ydata_profiling import ProfileReport
import great_expectations as gx
import sweetviz as sv


class ComprehensiveDataAnalyzer:
    """
    Comprehensive data analysis using three powerful libraries:
    1. ydata-profiling - Complete data understanding 
    2. Great Expectations - Data validation and quality
    3. Sweetviz - Beautiful visual data analysis
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "ai_data_analysis"
        self.temp_dir.mkdir(exist_ok=True)
    
    def analyze_dataset(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Run comprehensive analysis using all three libraries
        
        Args:
            df: The pandas DataFrame to analyze
            filename: Original filename for context
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Generate unique session ID for this analysis
            session_id = f"analysis_{hash(filename)}_{len(df)}"
            
            # Run all three analyses
            profiling_result = self._run_ydata_profiling(df, filename, session_id)
            expectations_result = self._run_great_expectations(df, filename, session_id)
            sweetviz_result = self._run_sweetviz(df, filename, session_id)
            
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
                "ydata_profiling": profiling_result,
                "great_expectations": expectations_result,
                "sweetviz": sweetviz_result,
                "executive_summary": self._generate_executive_summary(df, profiling_result, expectations_result, sweetviz_result)
            }
            
            return combined_result
            
        except Exception as e:
            return {
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None,
                "partial_results": True
            }
    
    def _run_ydata_profiling(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Run ydata-profiling analysis - Complete data understanding
        """
        try:
            # Generate profile report
            profile = ProfileReport(
                df,
                title=f"Healthcare Data Report - {filename}",
                explorative=True,
                dark_mode=False,
                minimal=False,
                html={
                    'style': {'full_width': True}
                }
            )
            
            # Save to temporary HTML file
            html_path = self.temp_dir / f"{session_id}_ydata_profile.html"
            profile.to_file(html_path)
            
            # Extract key insights from the profile
            dataset_stats = profile.get_description()
            
            # Create structured summary
            result = {
                "analysis_type": "ydata-profiling",
                "title": "Complete Data Understanding",
                "status": "success",
                "html_report_path": str(html_path),
                "key_insights": {
                    "data_types_detected": self._extract_data_types(dataset_stats),
                    "missing_values": self._extract_missing_values(dataset_stats),
                    "correlations": self._extract_correlations(dataset_stats),
                    "distributions": self._extract_distributions(dataset_stats),
                    "duplicates": self._extract_duplicates(dataset_stats),
                    "data_quality_warnings": self._extract_warnings(dataset_stats)
                },
                "detailed_stats": self._serialize_stats(dataset_stats),
                "recommendations": self._generate_profiling_recommendations(dataset_stats)
            }
            
            return result
            
        except Exception as e:
            return {
                "analysis_type": "ydata-profiling",
                "status": "error",
                "error": str(e),
                "message": "Failed to generate ydata-profiling report"
            }
    
    def _run_great_expectations(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Run Great Expectations analysis - Data validation and quality
        """
        try:
            # Create a data context
            context = gx.get_context()
            
            # Create a validator
            validator = context.sources.pandas_default.read_dataframe(df)
            
            # Run basic expectations suitable for healthcare data
            expectations_results = []
            
            # Basic data quality expectations
            try:
                # Check if data is not empty
                result = validator.expect_table_row_count_to_be_between(min_value=1)
                expectations_results.append({
                    "expectation": "Table not empty",
                    "success": result.success,
                    "details": result.result
                })
            except Exception as e:
                expectations_results.append({
                    "expectation": "Table not empty",
                    "success": False,
                    "error": str(e)
                })
            
            # Check column completeness for each column
            for column in df.columns:
                try:
                    if df[column].dtype in ['object', 'string']:
                        # For text columns - check for non-null values
                        result = validator.expect_column_values_to_not_be_null(column)
                        expectations_results.append({
                            "expectation": f"Column '{column}' completeness",
                            "success": result.success,
                            "details": {
                                "column": column,
                                "null_percentage": df[column].isnull().mean() * 100,
                                "data_type": "text"
                            }
                        })
                    elif df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                        # For numeric columns - check for valid ranges
                        result = validator.expect_column_values_to_not_be_null(column)
                        expectations_results.append({
                            "expectation": f"Column '{column}' numeric validity",
                            "success": result.success,
                            "details": {
                                "column": column,
                                "null_percentage": df[column].isnull().mean() * 100,
                                "min_value": float(df[column].min()) if not df[column].empty else None,
                                "max_value": float(df[column].max()) if not df[column].empty else None,
                                "data_type": "numeric"
                            }
                        })
                except Exception as e:
                    expectations_results.append({
                        "expectation": f"Column '{column}' validation",
                        "success": False,
                        "error": str(e)
                    })
            
            # Calculate overall data quality score
            successful_expectations = sum(1 for r in expectations_results if r.get('success', False))
            total_expectations = len(expectations_results)
            quality_score = (successful_expectations / total_expectations * 100) if total_expectations > 0 else 0
            
            result = {
                "analysis_type": "great-expectations",
                "title": "Data Quality Validation",
                "status": "success",
                "overall_quality_score": quality_score,
                "expectations_results": expectations_results,
                "summary": {
                    "total_expectations": total_expectations,
                    "successful_expectations": successful_expectations,
                    "failed_expectations": total_expectations - successful_expectations,
                    "quality_grade": self._get_quality_grade(quality_score)
                },
                "recommendations": self._generate_expectations_recommendations(expectations_results, quality_score)
            }
            
            return result
            
        except Exception as e:
            return {
                "analysis_type": "great-expectations",
                "status": "error",
                "error": str(e),
                "message": "Failed to run Great Expectations validation"
            }
    
    def _run_sweetviz(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Run Sweetviz analysis - Beautiful visual data analysis
        """
        try:
            # Generate Sweetviz report
            report = sv.analyze(df, target_feat=None)
            
            # Save to temporary HTML file
            html_path = self.temp_dir / f"{session_id}_sweetviz_report.html"
            report.show_html(str(html_path), open_browser=False, layout='vertical')
            
            # Extract key insights from Sweetviz analysis
            # Note: Sweetviz doesn't provide direct access to stats, so we'll calculate our own
            
            insights = {
                "categorical_analysis": self._analyze_categorical_columns(df),
                "numerical_analysis": self._analyze_numerical_columns(df),
                "data_overview": {
                    "total_features": len(df.columns),
                    "categorical_features": len(df.select_dtypes(include=['object', 'category']).columns),
                    "numerical_features": len(df.select_dtypes(include=[np.number]).columns),
                    "datetime_features": len(df.select_dtypes(include=['datetime64']).columns)
                },
                "quality_insights": {
                    "completeness_score": ((df.notna().sum().sum()) / (len(df) * len(df.columns)) * 100),
                    "duplicate_rows": df.duplicated().sum(),
                    "unique_value_ratios": {col: df[col].nunique() / len(df) for col in df.columns}
                }
            }
            
            result = {
                "analysis_type": "sweetviz",
                "title": "Visual Data Exploration",
                "status": "success",
                "html_report_path": str(html_path),
                "insights": insights,
                "recommendations": self._generate_sweetviz_recommendations(insights)
            }
            
            return result
            
        except Exception as e:
            return {
                "analysis_type": "sweetviz",
                "status": "error",
                "error": str(e),
                "message": "Failed to generate Sweetviz report"
            }
    
    def _analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical columns for Sweetviz insights"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        analysis = {}
        
        for col in categorical_cols:
            analysis[col] = {
                "unique_values": int(df[col].nunique()),
                "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "missing_percentage": float(df[col].isnull().mean() * 100),
                "cardinality": "high" if df[col].nunique() > len(df) * 0.5 else "low"
            }
        
        return analysis
    
    def _analyze_numerical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numerical columns for Sweetviz insights"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        analysis = {}
        
        for col in numerical_cols:
            analysis[col] = {
                "mean": float(df[col].mean()) if not df[col].empty else None,
                "median": float(df[col].median()) if not df[col].empty else None,
                "std": float(df[col].std()) if not df[col].empty else None,
                "min": float(df[col].min()) if not df[col].empty else None,
                "max": float(df[col].max()) if not df[col].empty else None,
                "missing_percentage": float(df[col].isnull().mean() * 100),
                "outliers_detected": self._detect_outliers(df[col])
            }
        
        return analysis
    
    def _detect_outliers(self, series: pd.Series) -> int:
        """Simple outlier detection using IQR method"""
        try:
            if series.dtype not in [np.number]:
                return 0
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return len(outliers)
        except:
            return 0
    
    def _extract_data_types(self, stats: Dict) -> Dict[str, Any]:
        """Extract data type information from ydata-profiling stats"""
        try:
            variables = stats.get('variables', {})
            data_types = {}
            
            for var_name, var_info in variables.items():
                data_types[var_name] = {
                    "type": var_info.get('type', 'unknown'),
                    "dtype": str(var_info.get('dtype', 'unknown'))
                }
            
            return data_types
        except:
            return {}
    
    def _extract_missing_values(self, stats: Dict) -> Dict[str, Any]:
        """Extract missing value information"""
        try:
            variables = stats.get('variables', {})
            missing_info = {}
            
            for var_name, var_info in variables.items():
                missing_count = var_info.get('n_missing', 0)
                total_count = var_info.get('count', 0)
                missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0
                
                missing_info[var_name] = {
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage
                }
            
            return missing_info
        except:
            return {}
    
    def _extract_correlations(self, stats: Dict) -> Dict[str, Any]:
        """Extract correlation information"""
        try:
            correlations = stats.get('correlations', {})
            return correlations
        except:
            return {}
    
    def _extract_distributions(self, stats: Dict) -> Dict[str, Any]:
        """Extract distribution information"""
        try:
            variables = stats.get('variables', {})
            distributions = {}
            
            for var_name, var_info in variables.items():
                if var_info.get('type') in ['Numeric', 'Integer']:
                    distributions[var_name] = {
                        "mean": var_info.get('mean'),
                        "std": var_info.get('std'),
                        "min": var_info.get('min'),
                        "max": var_info.get('max'),
                        "skewness": var_info.get('skewness')
                    }
            
            return distributions
        except:
            return {}
    
    def _extract_duplicates(self, stats: Dict) -> Dict[str, Any]:
        """Extract duplicate information"""
        try:
            table_stats = stats.get('table', {})
            return {
                "duplicate_rows": table_stats.get('n_duplicates', 0),
                "duplicate_percentage": (table_stats.get('n_duplicates', 0) / table_stats.get('n', 1) * 100)
            }
        except:
            return {"duplicate_rows": 0, "duplicate_percentage": 0}
    
    def _extract_warnings(self, stats: Dict) -> list:
        """Extract data quality warnings"""
        try:
            alerts = stats.get('alerts', [])
            warnings = []
            
            for alert in alerts:
                warnings.append({
                    "type": alert.get('alert_type', 'unknown'),
                    "message": alert.get('message', ''),
                    "column": alert.get('column_name', '')
                })
            
            return warnings
        except:
            return []
    
    def _serialize_stats(self, stats: Dict) -> Dict[str, Any]:
        """Serialize stats for JSON compatibility"""
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            return convert_types(stats)
        except:
            return {}
    
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
    
    def _generate_profiling_recommendations(self, stats: Dict) -> list:
        """Generate recommendations based on ydata-profiling results"""
        recommendations = []
        
        try:
            variables = stats.get('variables', {})
            
            # Check for high missing values
            for var_name, var_info in variables.items():
                missing_count = var_info.get('n_missing', 0)
                total_count = var_info.get('count', 0)
                missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0
                
                if missing_percentage > 50:
                    recommendations.append(f"⚠️ Column '{var_name}' has {missing_percentage:.1f}% missing values - consider imputation or removal")
                elif missing_percentage > 20:
                    recommendations.append(f"📊 Column '{var_name}' has {missing_percentage:.1f}% missing values - review data collection process")
            
            # Check for duplicates
            table_stats = stats.get('table', {})
            duplicate_percentage = (table_stats.get('n_duplicates', 0) / table_stats.get('n', 1) * 100)
            if duplicate_percentage > 5:
                recommendations.append(f"🔍 Dataset contains {duplicate_percentage:.1f}% duplicate rows - consider deduplication")
            
            # Check for highly correlated variables
            correlations = stats.get('correlations', {})
            if correlations:
                recommendations.append("📈 Review correlation matrix for multicollinearity issues in statistical modeling")
            
        except:
            recommendations.append("📋 Review the detailed profiling report for comprehensive insights")
        
        return recommendations
    
    def _generate_expectations_recommendations(self, results: list, quality_score: float) -> list:
        """Generate recommendations based on Great Expectations results"""
        recommendations = []
        
        if quality_score < 70:
            recommendations.append("🚨 Data quality score is below acceptable threshold - immediate attention required")
        elif quality_score < 85:
            recommendations.append("⚠️ Data quality needs improvement - review failed validations")
        else:
            recommendations.append("✅ Good data quality detected - suitable for analysis")
        
        # Analyze failed expectations
        failed_results = [r for r in results if not r.get('success', True)]
        
        if failed_results:
            recommendations.append(f"🔧 {len(failed_results)} validation(s) failed - review data collection and preprocessing")
        
        # Check for specific issues
        for result in results:
            if not result.get('success', True) and 'null' in result.get('expectation', '').lower():
                recommendations.append("📝 Missing value issues detected - implement data cleaning strategies")
        
        return recommendations
    
    def _generate_sweetviz_recommendations(self, insights: Dict) -> list:
        """Generate recommendations based on Sweetviz insights"""
        recommendations = []
        
        try:
            quality_insights = insights.get('quality_insights', {})
            completeness_score = quality_insights.get('completeness_score', 0)
            
            if completeness_score < 80:
                recommendations.append(f"📊 Data completeness is {completeness_score:.1f}% - consider data imputation strategies")
            
            duplicate_rows = quality_insights.get('duplicate_rows', 0)
            if duplicate_rows > 0:
                recommendations.append(f"🔍 Found {duplicate_rows} duplicate rows - consider deduplication")
            
            # Check cardinality issues
            categorical_analysis = insights.get('categorical_analysis', {})
            for col, analysis in categorical_analysis.items():
                if analysis.get('cardinality') == 'high':
                    recommendations.append(f"📈 Column '{col}' has high cardinality - consider grouping or encoding strategies")
            
            # Check for outliers
            numerical_analysis = insights.get('numerical_analysis', {})
            for col, analysis in numerical_analysis.items():
                outliers = analysis.get('outliers_detected', 0)
                if outliers > 0:
                    recommendations.append(f"📉 Column '{col}' has {outliers} potential outliers - review for data entry errors")
            
        except:
            recommendations.append("📋 Review the visual analysis report for detailed insights")
        
        return recommendations
    
    def _generate_executive_summary(self, df: pd.DataFrame, profiling_result: Dict, 
                                  expectations_result: Dict, sweetviz_result: Dict) -> Dict[str, Any]:
        """Generate an executive summary combining all three analyses"""
        
        # Overall data quality assessment
        quality_scores = []
        
        if expectations_result.get('status') == 'success':
            quality_scores.append(expectations_result.get('overall_quality_score', 0))
        
        if sweetviz_result.get('status') == 'success':
            completeness = sweetviz_result.get('insights', {}).get('quality_insights', {}).get('completeness_score', 0)
            quality_scores.append(completeness)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Key findings
        key_findings = []
        
        # From profiling
        if profiling_result.get('status') == 'success':
            missing_values = profiling_result.get('key_insights', {}).get('missing_values', {})
            high_missing_cols = [col for col, info in missing_values.items() 
                               if info.get('missing_percentage', 0) > 20]
            if high_missing_cols:
                key_findings.append(f"High missing values detected in {len(high_missing_cols)} columns")
        
        # From expectations
        if expectations_result.get('status') == 'success':
            failed_expectations = expectations_result.get('summary', {}).get('failed_expectations', 0)
            if failed_expectations > 0:
                key_findings.append(f"{failed_expectations} data quality validations failed")
        
        # From sweetviz
        if sweetviz_result.get('status') == 'success':
            duplicates = sweetviz_result.get('insights', {}).get('quality_insights', {}).get('duplicate_rows', 0)
            if duplicates > 0:
                key_findings.append(f"{duplicates} duplicate rows found")
        
        # Priority recommendations
        priority_recommendations = []
        
        for result in [profiling_result, expectations_result, sweetviz_result]:
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
                "ydata_profiling": profiling_result.get('status') == 'success',
                "great_expectations": expectations_result.get('status') == 'success',
                "sweetviz": sweetviz_result.get('status') == 'success'
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