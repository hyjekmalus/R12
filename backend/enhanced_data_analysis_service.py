"""
Enhanced Comprehensive Data Analysis Service
Integrating ydata-profiling, Great Expectations, and Sweetviz
for medical data with professional-grade profiling and validation
"""

import pandas as pd
import numpy as np
import io
import base64
import tempfile
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import warnings
import uuid
from datetime import datetime
import shutil

# Enhanced profiling libraries
from ydata_profiling import ProfileReport
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration
import sweetviz as sv

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
import logging
logging.getLogger('great_expectations').setLevel(logging.WARNING)


class EnhancedDataAnalyzer:
    """
    Enhanced comprehensive data analysis for medical statistics
    Integrates ydata-profiling, Great Expectations, and Sweetviz
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "enhanced_ai_data_analysis"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different report types
        self.profiling_dir = self.temp_dir / "profiling_reports"
        self.validation_dir = self.temp_dir / "validation_reports" 
        self.sweetviz_dir = self.temp_dir / "sweetviz_reports"
        
        for dir_path in [self.profiling_dir, self.validation_dir, self.sweetviz_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def analyze_dataset(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Run enhanced comprehensive analysis using professional-grade tools
        
        Args:
            df: The pandas DataFrame to analyze
            filename: Original filename for context
            
        Returns:
            Dictionary containing all analysis results with HTML reports
        """
        try:
            # Generate unique session ID for this analysis
            session_id = f"enhanced_analysis_{uuid.uuid4().hex[:8]}_{len(df)}"
            
            print(f"Starting enhanced analysis for {filename} with session {session_id}")
            
            # Run all enhanced analyses
            profiling_result = self._run_ydata_profiling(df, filename, session_id)
            validation_result = self._run_great_expectations_validation(df, filename, session_id)
            exploration_result = self._run_sweetviz_exploration(df, filename, session_id)
            
            # Combine results with enhanced summary
            combined_result = {
                "session_id": session_id,
                "filename": filename,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "dataset_info": {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "memory_usage": f"{df.memory_usage().sum() / 1024 / 1024:.2f} MB",
                    "data_quality_score": self._calculate_overall_quality_score(df)
                },
                "enhanced_profiling": profiling_result,
                "medical_validation": validation_result,
                "exploratory_analysis": exploration_result,
                "executive_summary": self._generate_enhanced_executive_summary(df, profiling_result, validation_result, exploration_result),
                "ai_context_summary": self._generate_ai_context_summary(df, profiling_result, validation_result, exploration_result)
            }
            
            print(f"Enhanced analysis completed successfully for {filename}")
            return combined_result
            
        except Exception as e:
            print(f"Enhanced analysis error for {filename}: {str(e)}")
            return {
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None,
                "partial_results": True,
                "fallback_message": "Enhanced profiling encountered an issue, falling back to basic analysis"
            }
    
    def _run_ydata_profiling(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive data profiling report using ydata-profiling
        """
        try:
            print(f"Generating ydata-profiling report for {filename}")
            
            # Configure profile for medical data
            profile_config = {
                "title": f"Medical Data Profiling Report - {filename}",
                "dataset": {
                    "description": f"Comprehensive profiling analysis of medical dataset: {filename}"
                },
                "variables": {
                    "descriptions": {},
                },
                "correlations": {
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                    "kendall": {"calculate": False},  # Skip kendall for performance
                    "phi_k": {"calculate": True}
                },
                "missing_diagrams": {
                    "bar": True,
                    "matrix": True,
                    "heatmap": True
                },
                "interactions": {
                    "continuous": True,
                    "targets": []
                },
                "samples": {
                    "head": 10,
                    "tail": 10
                }
            }
            
            # Generate profile report
            profile = ProfileReport(
                df, 
                title=f"Medical Data Analysis - {filename}",
                explorative=True,
                config_file=None,
                **profile_config
            )
            
            # Save HTML report
            report_filename = f"profiling_report_{session_id}.html"
            report_path = self.profiling_dir / report_filename
            profile.to_file(report_path)
            
            # Extract key insights from the profile
            profile_json = profile.to_json()
            profile_data = json.loads(profile_json)
            
            # Extract summary statistics
            summary_stats = self._extract_profiling_insights(profile_data, df)
            
            result = {
                "analysis_type": "ydata-profiling",
                "title": "📊 Comprehensive Data Profiling Report",
                "status": "success",
                "report_path": str(report_path),
                "report_filename": report_filename,
                "key_insights": summary_stats,
                "medical_context": self._add_medical_context_to_profiling(summary_stats, df),
                "html_summary": self._generate_profiling_html_summary(summary_stats, filename)
            }
            
            print(f"ydata-profiling completed for {filename}")
            return result
            
        except Exception as e:
            print(f"ydata-profiling error: {str(e)}")
            return {
                "analysis_type": "ydata-profiling",
                "status": "error",
                "error": str(e),
                "message": "Failed to generate comprehensive profiling report"
            }
    
    def _run_great_expectations_validation(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Run data validation using Great Expectations with medical data focus
        """
        try:
            print(f"Running Great Expectations validation for {filename}")
            
            # Create a temporary great expectations context
            context = gx.get_context()
            
            # Create a pandas datasource
            datasource_name = f"medical_data_{session_id}"
            datasource = context.sources.add_pandas(datasource_name)
            
            # Add data asset
            data_asset_name = f"dataset_{session_id}"
            data_asset = datasource.add_dataframe_asset(data_asset_name)
            
            # Create batch request
            batch_request = data_asset.build_batch_request(dataframe=df)
            
            # Create expectation suite for medical data
            suite_name = f"medical_validation_suite_{session_id}"
            suite = context.add_expectation_suite(expectation_suite_name=suite_name)
            
            # Add medical-specific expectations
            medical_expectations = self._create_medical_expectations(df)
            
            for expectation in medical_expectations:
                suite.add_expectation(expectation)
            
            # Run validation
            validator = context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            # Execute validation
            validation_result = validator.validate()
            
            # Generate validation report
            report_filename = f"validation_report_{session_id}.html"
            report_path = self.validation_dir / report_filename
            
            # Convert validation results to readable format
            validation_summary = self._process_validation_results(validation_result)
            
            # Generate HTML report
            html_report = self._generate_validation_html_report(validation_summary, filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            result = {
                "analysis_type": "great-expectations",
                "title": "🔍 Medical Data Validation Report",
                "status": "success",
                "report_path": str(report_path),
                "report_filename": report_filename,
                "validation_summary": validation_summary,
                "overall_success": validation_result.success,
                "medical_compliance": self._assess_medical_compliance(validation_summary),
                "html_summary": self._generate_validation_html_summary(validation_summary, filename)
            }
            
            print(f"Great Expectations validation completed for {filename}")
            return result
            
        except Exception as e:
            print(f"Great Expectations error: {str(e)}")
            return {
                "analysis_type": "great-expectations", 
                "status": "error",
                "error": str(e),
                "message": "Failed to run data validation checks"
            }
    
    def _run_sweetviz_exploration(self, df: pd.DataFrame, filename: str, session_id: str) -> Dict[str, Any]:
        """
        Generate exploratory data analysis using Sweetviz
        """
        try:
            print(f"Generating Sweetviz EDA report for {filename}")
            
            # Configure Sweetviz for medical data
            config = sv.FeatureConfig(skip="", force_text=[], force_num=[])
            
            # Generate analysis report
            report = sv.analyze(
                source=df,
                target_feat=None,  # Can be set to a specific target column if needed
                feat_cfg=config,
                pairwise_analysis="auto"
            )
            
            # Save HTML report
            report_filename = f"sweetviz_report_{session_id}.html"
            report_path = self.sweetviz_dir / report_filename
            report.show_html(str(report_path), open_browser=False, layout='vertical')
            
            # Extract insights from the analysis
            sweetviz_insights = self._extract_sweetviz_insights(df)
            
            result = {
                "analysis_type": "sweetviz",
                "title": "📈 Exploratory Data Analysis Report", 
                "status": "success",
                "report_path": str(report_path),
                "report_filename": report_filename,
                "key_insights": sweetviz_insights,
                "medical_insights": self._add_medical_context_to_eda(sweetviz_insights, df),
                "html_summary": self._generate_sweetviz_html_summary(sweetviz_insights, filename)
            }
            
            print(f"Sweetviz EDA completed for {filename}")
            return result
            
        except Exception as e:
            print(f"Sweetviz error: {str(e)}")
            return {
                "analysis_type": "sweetviz",
                "status": "error", 
                "error": str(e),
                "message": "Failed to generate exploratory data analysis report"
            }
    
    def _create_medical_expectations(self, df: pd.DataFrame) -> List[ExpectationConfiguration]:
        """
        Create Great Expectations specific to medical data validation
        """
        expectations = []
        
        # Basic data integrity expectations
        expectations.append(
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={"min_value": 1, "max_value": 1000000}
            )
        )
        
        expectations.append(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_be_between", 
                kwargs={"min_value": 1, "max_value": 1000}
            )
        )
        
        # Column-specific medical expectations
        for column in df.columns:
            col_data = df[column]
            
            # Basic column existence
            expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": column}
                )
            )
            
            # Missing value expectations (critical for medical data)
            missing_percentage = col_data.isnull().mean() * 100
            if missing_percentage > 50:
                # High missing data warning
                expectations.append(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_not_be_null",
                        kwargs={
                            "column": column,
                            "mostly": 0.1  # At least 10% should not be null
                        }
                    )
                )
            else:
                # Normal missing data expectation
                expectations.append(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_not_be_null",
                        kwargs={
                            "column": column,
                            "mostly": 0.8  # At least 80% should not be null
                        }
                    )
                )
            
            # Numeric column expectations (common in medical data)
            if pd.api.types.is_numeric_dtype(col_data):
                # Check for reasonable ranges (avoid extreme outliers)
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                if not col_data.empty and not (pd.isna(lower_bound) or pd.isna(upper_bound)):
                    expectations.append(
                        ExpectationConfiguration(
                            expectation_type="expect_column_values_to_be_between",
                            kwargs={
                                "column": column,
                                "min_value": float(lower_bound) if not pd.isna(lower_bound) else None,
                                "max_value": float(upper_bound) if not pd.isna(upper_bound) else None,
                                "mostly": 0.95
                            }
                        )
                    )
                
                # Age-specific expectations (common medical variable)
                if any(age_term in column.lower() for age_term in ['age', 'años', 'year']):
                    expectations.append(
                        ExpectationConfiguration(
                            expectation_type="expect_column_values_to_be_between",
                            kwargs={
                                "column": column,
                                "min_value": 0,
                                "max_value": 120,
                                "mostly": 0.98
                            }
                        )
                    )
            
            # Categorical column expectations
            elif pd.api.types.is_object_dtype(col_data):
                # Check for reasonable number of unique values
                unique_count = col_data.nunique()
                if unique_count > 0:
                    # Gender/sex columns (common in medical data)
                    if any(gender_term in column.lower() for gender_term in ['gender', 'sex', 'sexo', 'genero']):
                        common_gender_values = ['male', 'female', 'M', 'F', 'masculino', 'femenino', '1', '0']
                        expectations.append(
                            ExpectationConfiguration(
                                expectation_type="expect_column_values_to_be_in_set",
                                kwargs={
                                    "column": column,
                                    "value_set": common_gender_values,
                                    "mostly": 0.9
                                }
                            )
                        )
                    
                    # ID columns should be unique
                    if any(id_term in column.lower() for id_term in ['id', 'patient', 'subject', 'participante']):
                        expectations.append(
                            ExpectationConfiguration(
                                expectation_type="expect_column_values_to_be_unique",
                                kwargs={"column": column}
                            )
                        )
        
        return expectations
    
    def _process_validation_results(self, validation_result) -> Dict[str, Any]:
        """
        Process Great Expectations validation results into readable format
        """
        results = {
            "overall_success": validation_result.success,
            "total_expectations": len(validation_result.results),
            "successful_expectations": sum(1 for r in validation_result.results if r.success),
            "failed_expectations": sum(1 for r in validation_result.results if not r.success),
            "expectation_details": []
        }
        
        for result in validation_result.results:
            expectation_detail = {
                "expectation_type": result.expectation_config.expectation_type,
                "success": result.success,
                "column": result.expectation_config.kwargs.get("column", "table"),
                "details": self._format_expectation_result(result)
            }
            results["expectation_details"].append(expectation_detail)
        
        # Calculate quality score
        if results["total_expectations"] > 0:
            results["quality_score"] = (results["successful_expectations"] / results["total_expectations"]) * 100
        else:
            results["quality_score"] = 100
        
        return results
    
    def _format_expectation_result(self, result) -> str:
        """
        Format expectation result into human-readable description
        """
        expectation_type = result.expectation_config.expectation_type
        kwargs = result.expectation_config.kwargs
        
        if expectation_type == "expect_column_values_to_not_be_null":
            column = kwargs.get("column", "unknown")
            mostly = kwargs.get("mostly", 1.0)
            return f"Column '{column}' should have at least {mostly*100:.0f}% non-null values"
        
        elif expectation_type == "expect_column_values_to_be_between":
            column = kwargs.get("column", "unknown")
            min_val = kwargs.get("min_value")
            max_val = kwargs.get("max_value")
            mostly = kwargs.get("mostly", 1.0)
            return f"Column '{column}' values should be between {min_val} and {max_val} for {mostly*100:.0f}% of records"
        
        elif expectation_type == "expect_column_values_to_be_unique":
            column = kwargs.get("column", "unknown")
            return f"Column '{column}' should contain unique values"
        
        elif expectation_type == "expect_column_values_to_be_in_set":
            column = kwargs.get("column", "unknown")
            value_set = kwargs.get("value_set", [])
            mostly = kwargs.get("mostly", 1.0)
            return f"Column '{column}' should contain values from {value_set} for {mostly*100:.0f}% of records"
        
        else:
            return f"Expectation: {expectation_type}"
    
    def _extract_profiling_insights(self, profile_data: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract key insights from ydata-profiling results
        """
        try:
            insights = {
                "dataset_size": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_usage_mb": df.memory_usage().sum() / 1024 / 1024
                },
                "data_types": {
                    "numeric": len(df.select_dtypes(include=[np.number]).columns),
                    "categorical": len(df.select_dtypes(include=['object']).columns),
                    "datetime": len(df.select_dtypes(include=['datetime64']).columns)
                },
                "missing_data": {
                    "total_missing": df.isnull().sum().sum(),
                    "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
                    "columns_with_missing": df.isnull().any().sum(),
                    "complete_rows": len(df.dropna())
                },
                "duplicates": {
                    "duplicate_rows": df.duplicated().sum(),
                    "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
                }
            }
            
            # Add correlation insights for numeric data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j and abs(corr_matrix.loc[col1, col2]) > 0.7:
                            high_corr_pairs.append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": float(corr_matrix.loc[col1, col2])
                            })
                
                insights["correlations"] = {
                    "high_correlation_pairs": high_corr_pairs,
                    "total_numeric_columns": len(numeric_cols)
                }
            
            return insights
            
        except Exception as e:
            return {"error": str(e), "message": "Failed to extract profiling insights"}
    
    def _extract_sweetviz_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract key insights for Sweetviz analysis
        """
        insights = {
            "feature_analysis": {},
            "distribution_insights": {},
            "data_quality": {}
        }
        
        # Analyze each column
        for column in df.columns:
            col_data = df[column]
            
            if pd.api.types.is_numeric_dtype(col_data):
                insights["feature_analysis"][column] = {
                    "type": "numeric",
                    "unique_values": int(col_data.nunique()),
                    "missing_count": int(col_data.isnull().sum()),
                    "mean": float(col_data.mean()) if not col_data.empty else None,
                    "std": float(col_data.std()) if not col_data.empty else None,
                    "skewness": float(col_data.skew()) if not col_data.empty else None
                }
            else:
                insights["feature_analysis"][column] = {
                    "type": "categorical",
                    "unique_values": int(col_data.nunique()),
                    "missing_count": int(col_data.isnull().sum()),
                    "most_frequent": str(col_data.mode().iloc[0]) if not col_data.empty and len(col_data.mode()) > 0 else None
                }
        
        return insights
    
    def _calculate_overall_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate an overall data quality score (0-100)
        """
        scores = []
        
        # Completeness score (missing data)
        completeness = (1 - (df.isnull().sum().sum() / df.size)) * 100
        scores.append(completeness)
        
        # Uniqueness score (for potential ID columns)
        uniqueness_scores = []
        for col in df.columns:
            if df[col].dtype == 'object' and any(id_term in col.lower() for id_term in ['id', 'patient']):
                uniqueness = (df[col].nunique() / len(df)) * 100
                uniqueness_scores.append(uniqueness)
        
        if uniqueness_scores:
            scores.append(np.mean(uniqueness_scores))
        
        # Consistency score (no extreme outliers in numeric data)
        consistency_scores = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if not df[col].empty:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outliers = df[(df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)][col]
                    consistency = (1 - len(outliers) / len(df)) * 100
                    consistency_scores.append(consistency)
        
        if consistency_scores:
            scores.append(np.mean(consistency_scores))
        
        return np.mean(scores) if scores else 75.0  # Default moderate score
    
    def _add_medical_context_to_profiling(self, insights: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Add medical-specific context to profiling insights
        """
        medical_context = {
            "patient_data_assessment": {},
            "medical_variable_detection": {},
            "clinical_data_quality": {}
        }
        
        # Detect common medical variables
        medical_variables = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['age', 'edad', 'años']):
                medical_variables.append({"column": col, "type": "age", "importance": "high"})
            elif any(term in col_lower for term in ['gender', 'sex', 'sexo', 'genero']):
                medical_variables.append({"column": col, "type": "gender", "importance": "high"})
            elif any(term in col_lower for term in ['weight', 'peso', 'kg']):
                medical_variables.append({"column": col, "type": "weight", "importance": "medium"})
            elif any(term in col_lower for term in ['height', 'altura', 'cm']):
                medical_variables.append({"column": col, "type": "height", "importance": "medium"})
            elif any(term in col_lower for term in ['pressure', 'presion', 'bp', 'systolic', 'diastolic']):
                medical_variables.append({"column": col, "type": "blood_pressure", "importance": "high"})
            elif any(term in col_lower for term in ['glucose', 'glucosa', 'sugar']):
                medical_variables.append({"column": col, "type": "glucose", "importance": "high"})
        
        medical_context["medical_variable_detection"] = {
            "detected_variables": medical_variables,
            "medical_relevance_score": len(medical_variables) / len(df.columns) * 100
        }
        
        # Assess clinical data quality
        missing_in_critical = []
        for var in medical_variables:
            if var["importance"] == "high":
                col = var["column"]
                missing_pct = df[col].isnull().mean() * 100
                if missing_pct > 10:  # More than 10% missing in critical medical variables
                    missing_in_critical.append({"column": col, "missing_percentage": missing_pct})
        
        medical_context["clinical_data_quality"] = {
            "critical_variables_with_missing": missing_in_critical,
            "clinical_readiness_score": max(0, 100 - len(missing_in_critical) * 20)
        }
        
        return medical_context
    
    def _add_medical_context_to_eda(self, insights: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Add medical-specific context to EDA insights
        """
        medical_eda = {
            "clinical_patterns": {},
            "medical_anomalies": {},
            "research_recommendations": {}
        }
        
        # Look for clinical patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Age distribution analysis (if age column exists)
        age_cols = [col for col in df.columns if any(term in col.lower() for term in ['age', 'edad'])]
        if age_cols:
            age_col = age_cols[0]
            age_data = df[age_col].dropna()
            if not age_data.empty:
                medical_eda["clinical_patterns"]["age_distribution"] = {
                    "mean_age": float(age_data.mean()),
                    "age_range": f"{age_data.min():.0f} - {age_data.max():.0f}",
                    "elderly_percentage": float((age_data >= 65).mean() * 100),
                    "pediatric_percentage": float((age_data < 18).mean() * 100)
                }
        
        # Research recommendations based on data characteristics
        recommendations = []
        if len(numeric_cols) >= 3:
            recommendations.append("Dataset suitable for correlation analysis and multivariate statistics")
        if len(df) >= 30:
            recommendations.append("Sample size adequate for most statistical tests")
        if len(df) >= 100:
            recommendations.append("Sample size good for machine learning approaches")
        
        medical_eda["research_recommendations"]["statistical_approaches"] = recommendations
        
        return medical_eda
    
    def _assess_medical_compliance(self, validation_summary: Dict) -> Dict[str, Any]:
        """
        Assess compliance with medical data standards
        """
        compliance = {
            "overall_score": validation_summary.get("quality_score", 0),
            "medical_standards": {},
            "recommendations": []
        }
        
        # Medical data standards assessment
        if validation_summary.get("quality_score", 0) >= 90:
            compliance["medical_standards"]["grade"] = "Excellent"
            compliance["recommendations"].append("Data meets high medical research standards")
        elif validation_summary.get("quality_score", 0) >= 80:
            compliance["medical_standards"]["grade"] = "Good" 
            compliance["recommendations"].append("Data suitable for most medical analyses")
        elif validation_summary.get("quality_score", 0) >= 70:
            compliance["medical_standards"]["grade"] = "Acceptable"
            compliance["recommendations"].append("Consider data cleaning before complex analyses")
        else:
            compliance["medical_standards"]["grade"] = "Needs Improvement"
            compliance["recommendations"].append("Significant data quality issues detected - thorough cleaning recommended")
        
        return compliance
    
    def _generate_enhanced_executive_summary(self, df: pd.DataFrame, profiling: Dict, validation: Dict, exploration: Dict) -> str:
        """
        Generate an enhanced executive summary combining all analyses
        """
        summary_parts = []
        
        # Dataset overview
        summary_parts.append(f"**📊 Dataset Overview:**")
        summary_parts.append(f"- **Size:** {len(df):,} patients/records × {len(df.columns)} variables")
        summary_parts.append(f"- **Data Quality Score:** {self._calculate_overall_quality_score(df):.1f}/100")
        
        # Data completeness
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        summary_parts.append(f"- **Data Completeness:** {100-missing_pct:.1f}% complete")
        
        # Medical relevance
        medical_vars = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['age', 'gender', 'sex', 'weight', 'height', 'pressure', 'glucose']):
                medical_vars.append(col)
        
        if medical_vars:
            summary_parts.append(f"- **Medical Variables Detected:** {len(medical_vars)} ({', '.join(medical_vars[:3])}{'...' if len(medical_vars) > 3 else ''})")
        
        # Validation results
        if validation.get("status") == "success":
            validation_summary = validation.get("validation_summary", {})
            quality_score = validation_summary.get("quality_score", 0)
            summary_parts.append(f"- **Validation Score:** {quality_score:.1f}% of quality checks passed")
        
        # Key recommendations
        summary_parts.append(f"\n**🎯 Key Recommendations:**")
        
        if missing_pct > 20:
            summary_parts.append(f"- ⚠️ High missing data ({missing_pct:.1f}%) - consider imputation strategies")
        
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            summary_parts.append(f"- ⚠️ {dup_count} duplicate records detected - review for data entry errors")
        
        if len(df) < 30:
            summary_parts.append(f"- ⚠️ Small sample size (n={len(df)}) - consider statistical power implications")
        
        summary_parts.append(f"- ✅ Data is ready for AI-powered statistical analysis")
        summary_parts.append(f"- ✅ Comprehensive profiling reports generated for detailed exploration")
        
        return "\n".join(summary_parts)
    
    def _generate_ai_context_summary(self, df: pd.DataFrame, profiling: Dict, validation: Dict, exploration: Dict) -> Dict[str, Any]:
        """
        Generate structured summary for AI context enhancement
        """
        ai_context = {
            "dataset_characteristics": {
                "size": {"rows": len(df), "columns": len(df.columns)},
                "data_types": {
                    "numeric_variables": len(df.select_dtypes(include=[np.number]).columns),
                    "categorical_variables": len(df.select_dtypes(include=['object']).columns),
                    "variable_names": df.columns.tolist()
                },
                "data_quality": {
                    "overall_score": self._calculate_overall_quality_score(df),
                    "missing_data_percentage": (df.isnull().sum().sum() / df.size) * 100,
                    "duplicate_records": int(df.duplicated().sum())
                }
            },
            "medical_context": {
                "detected_medical_variables": [],
                "clinical_relevance_indicators": [],
                "research_suitability": []
            },
            "analysis_recommendations": {
                "suitable_statistical_tests": [],
                "visualization_suggestions": [],
                "data_preprocessing_needs": []
            }
        }
        
        # Detect medical variables for AI context
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['age', 'edad']):
                ai_context["medical_context"]["detected_medical_variables"].append({"variable": col, "type": "demographic", "importance": "high"})
            elif any(term in col_lower for term in ['gender', 'sex', 'sexo']):
                ai_context["medical_context"]["detected_medical_variables"].append({"variable": col, "type": "demographic", "importance": "high"})
            elif any(term in col_lower for term in ['weight', 'height', 'bmi']):
                ai_context["medical_context"]["detected_medical_variables"].append({"variable": col, "type": "anthropometric", "importance": "medium"})
            elif any(term in col_lower for term in ['pressure', 'glucose', 'cholesterol']):
                ai_context["medical_context"]["detected_medical_variables"].append({"variable": col, "type": "clinical_measure", "importance": "high"})
        
        # Add analysis recommendations based on data characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            ai_context["analysis_recommendations"]["suitable_statistical_tests"].append("Correlation analysis")
            ai_context["analysis_recommendations"]["visualization_suggestions"].append("Correlation heatmap")
        
        if len(numeric_cols) >= 1 and len(df.select_dtypes(include=['object']).columns) >= 1:
            ai_context["analysis_recommendations"]["suitable_statistical_tests"].append("Group comparisons (t-tests, ANOVA)")
            ai_context["analysis_recommendations"]["visualization_suggestions"].append("Box plots by groups")
        
        if len(df) >= 100:
            ai_context["analysis_recommendations"]["suitable_statistical_tests"].append("Machine learning approaches")
            ai_context["medical_context"]["research_suitability"].append("Suitable for predictive modeling")
        
        # Data preprocessing recommendations
        if (df.isnull().sum().sum() / df.size) * 100 > 5:
            ai_context["analysis_recommendations"]["data_preprocessing_needs"].append("Missing data handling")
        
        if df.duplicated().any():
            ai_context["analysis_recommendations"]["data_preprocessing_needs"].append("Duplicate removal")
        
        return ai_context
    
    def _generate_profiling_html_summary(self, insights: Dict, filename: str) -> str:
        """Generate HTML summary for profiling results"""
        html = f"""
        <div class="profiling-summary bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <h3 class="text-lg font-semibold text-blue-800 mb-3">📊 Data Profiling Summary - {filename}</h3>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p><strong>Dataset Size:</strong> {insights.get('dataset_size', {}).get('rows', 'N/A'):,} rows × {insights.get('dataset_size', {}).get('columns', 'N/A')} columns</p>
                    <p><strong>Memory Usage:</strong> {insights.get('dataset_size', {}).get('memory_usage_mb', 0):.1f} MB</p>
                </div>
                <div>
                    <p><strong>Missing Data:</strong> {insights.get('missing_data', {}).get('missing_percentage', 0):.1f}%</p>
                    <p><strong>Duplicate Rows:</strong> {insights.get('duplicates', {}).get('duplicate_rows', 0)}</p>
                </div>
            </div>
        </div>
        """
        return html
    
    def _generate_validation_html_summary(self, validation_summary: Dict, filename: str) -> str:
        """Generate HTML summary for validation results"""
        quality_score = validation_summary.get('quality_score', 0)
        color_class = "green" if quality_score >= 80 else "yellow" if quality_score >= 60 else "red"
        
        html = f"""
        <div class="validation-summary bg-{color_class}-50 border border-{color_class}-200 rounded-lg p-4 mb-4">
            <h3 class="text-lg font-semibold text-{color_class}-800 mb-3">🔍 Validation Summary - {filename}</h3>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p><strong>Quality Score:</strong> {quality_score:.1f}%</p>
                    <p><strong>Total Checks:</strong> {validation_summary.get('total_expectations', 0)}</p>
                </div>
                <div>
                    <p><strong>Passed:</strong> {validation_summary.get('successful_expectations', 0)}</p>
                    <p><strong>Failed:</strong> {validation_summary.get('failed_expectations', 0)}</p>
                </div>
            </div>
        </div>
        """
        return html
    
    def _generate_sweetviz_html_summary(self, insights: Dict, filename: str) -> str:
        """Generate HTML summary for Sweetviz results"""
        total_features = len(insights.get('feature_analysis', {}))
        numeric_features = sum(1 for f in insights.get('feature_analysis', {}).values() if f.get('type') == 'numeric')
        
        html = f"""
        <div class="sweetviz-summary bg-purple-50 border border-purple-200 rounded-lg p-4 mb-4">
            <h3 class="text-lg font-semibold text-purple-800 mb-3">📈 EDA Summary - {filename}</h3>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p><strong>Total Features:</strong> {total_features}</p>
                    <p><strong>Numeric Features:</strong> {numeric_features}</p>
                </div>
                <div>
                    <p><strong>Categorical Features:</strong> {total_features - numeric_features}</p>
                    <p><strong>Analysis Type:</strong> Exploratory Data Analysis</p>
                </div>
            </div>
        </div>
        """
        return html
    
    def _generate_validation_html_report(self, validation_summary: Dict, filename: str) -> str:
        """Generate a complete HTML validation report"""
        quality_score = validation_summary.get('quality_score', 0)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical Data Validation Report - {filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metric {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .pass {{ color: #28a745; }}
                .fail {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .expectation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Medical Data Validation Report</h1>
                <h2>{filename}</h2>
                <p><strong>Overall Quality Score:</strong> {quality_score:.1f}%</p>
            </div>
            
            <div class="metric">
                <h3>📊 Validation Summary</h3>
                <p><strong>Total Expectations:</strong> {validation_summary.get('total_expectations', 0)}</p>
                <p><strong>Successful:</strong> <span class="pass">{validation_summary.get('successful_expectations', 0)}</span></p>
                <p><strong>Failed:</strong> <span class="fail">{validation_summary.get('failed_expectations', 0)}</span></p>
            </div>
            
            <div class="metric">
                <h3>📋 Detailed Results</h3>
        """
        
        for expectation in validation_summary.get('expectation_details', []):
            status_class = "pass" if expectation['success'] else "fail"
            status_text = "✅ PASS" if expectation['success'] else "❌ FAIL"
            
            html += f"""
                <div class="expectation">
                    <p><strong class="{status_class}">{status_text}</strong></p>
                    <p><strong>Column:</strong> {expectation.get('column', 'N/A')}</p>
                    <p><strong>Check:</strong> {expectation.get('details', 'N/A')}</p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B" 
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"