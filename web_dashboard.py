"""
OWLBAN GROUP AI Web Dashboard
Streamlit-based web interface for monitoring and controlling AI systems
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
import json
from typing import Dict, List, Optional, Any

# Import AI systems
try:
    from combined_nim_owlban_ai import CombinedSystem
    combined_system_available = True
except ImportError:
    combined_system_available = False

try:
    from new_products.revenue_optimizer import NVIDIARevenueOptimizer
    from combined_nim_owlban_ai.nim import NimManager
    revenue_optimizer_available = True
except ImportError:
    revenue_optimizer_available = False

try:
    from database_manager import DatabaseManager
    database_available = True
except ImportError:
    database_available = False

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="OWLBAN GROUP AI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class AIDashboard:
    """OWLBAN GROUP AI Dashboard"""

    def __init__(self):
        self.api_url = API_BASE_URL
        self.db_manager = DatabaseManager() if database_available else None

        # Initialize AI systems
        self.combined_system = CombinedSystem() if combined_system_available else None
        self.revenue_optimizer = None
        self.nim_manager = None

        if revenue_optimizer_available:
            self.nim_manager = NimManager()
            self.nim_manager.initialize()
            self.revenue_optimizer = NVIDIARevenueOptimizer(self.nim_manager)

    def run(self):
        """Run the dashboard"""
        st.markdown('<div class="main-header">🚀 OWLBAN GROUP AI Dashboard</div>', unsafe_allow_html=True)

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Overview", "AI Inference", "Revenue Optimization", "GPU Monitoring",
             "Quantum AI", "Database", "System Health", "Settings"]
        )

        # Main content
        if page == "Overview":
            self.show_overview()
        elif page == "AI Inference":
            self.show_inference()
        elif page == "Revenue Optimization":
            self.show_revenue_optimization()
        elif page == "GPU Monitoring":
            self.show_gpu_monitoring()
        elif page == "Quantum AI":
            self.show_quantum_ai()
        elif page == "Database":
            self.show_database()
        elif page == "System Health":
            self.show_system_health()
        elif page == "Settings":
            self.show_settings()

    def show_overview(self):
        """Show dashboard overview"""
        st.header("System Overview")

        # System status
        status = self.get_system_status()
        if status:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Combined System", "Active" if status.get('services', {}).get('combined_system') else "Inactive")

            with col2:
                st.metric("Revenue Optimizer", "Active" if status.get('services', {}).get('revenue_optimizer') else "Inactive")

            with col3:
                st.metric("RL Agent", "Active" if status.get('services', {}).get('rl_agent') else "Inactive")

            with col4:
                st.metric("NIM Manager", "Active" if status.get('services', {}).get('nim_manager') else "Inactive")

        # GPU status
        gpu_status = self.get_gpu_status()
        if gpu_status:
            st.subheader("GPU Resources")
            gpu_df = pd.DataFrame.from_dict(gpu_status, orient='index', columns=['Value'])
            st.dataframe(gpu_df)

        # Recent predictions
        if self.db_manager:
            st.subheader("Recent AI Predictions")
            predictions = self.db_manager.get_predictions(limit=10)
            if predictions:
                pred_df = pd.DataFrame(predictions)
                st.dataframe(pred_df[['model_name', 'confidence', 'timestamp']])

    def show_inference(self):
        """Show AI inference interface"""
        st.header("AI Inference")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Data")
            input_data = {}
            input_data['feature1'] = st.slider("Feature 1", 0.0, 1.0, 0.5)
            input_data['feature2'] = st.slider("Feature 2", 0.0, 1.0, 0.3)
            input_data['feature3'] = st.slider("Feature 3", 0.0, 1.0, 0.7)

        with col2:
            st.subheader("Inference Result")
            if st.button("Run Inference"):
                if self.combined_system:
                    try:
                        result = self.combined_system.run_inference(input_data)
                        st.success("Inference completed!")
                        st.json(result)

                        # Save to database
                        if self.db_manager:
                            self.db_manager.save_prediction(
                                "dashboard_inference",
                                input_data,
                                result,
                                result.get('confidence', 0.5)
                            )
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                else:
                    st.error("Combined system not available")

    def show_revenue_optimization(self):
        """Show revenue optimization interface"""
        st.header("Revenue Optimization")

        if not self.revenue_optimizer:
            st.error("Revenue optimizer not available")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Optimization Parameters")
            iterations = st.slider("Iterations", 1, 100, 10)

            if st.button("Start Optimization"):
                try:
                    self.revenue_optimizer.optimize_revenue(iterations)
                    st.success(f"Optimization completed with {iterations} iterations!")
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

        with col2:
            st.subheader("Current Performance")
            try:
                profit = self.revenue_optimizer.get_current_profit()
                st.metric("Current Profit", f"${profit:.2f}")

                # Quantum portfolio
                if st.button("Optimize Quantum Portfolio"):
                    portfolio = self.revenue_optimizer.optimize_quantum_portfolio()
                    st.json(portfolio.__dict__)

            except Exception as e:
                st.error(f"Failed to get performance: {e}")

    def show_gpu_monitoring(self):
        """Show GPU monitoring dashboard"""
        st.header("GPU Monitoring")

        gpu_status = self.get_gpu_status()
        if gpu_status:
            # Create GPU metrics chart
            gpu_data = []
            for key, value in gpu_status.items():
                if "Usage" in key and "%" in str(value):
                    gpu_data.append({
                        'GPU': key.replace('_Usage', '').replace('_', ' '),
                        'Usage': float(str(value).strip('%'))
                    })

            if gpu_data:
                df = pd.DataFrame(gpu_data)
                fig = px.bar(df, x='GPU', y='Usage', title='GPU Utilization')
                st.plotly_chart(fig)

            # Raw GPU data
            st.subheader("Detailed GPU Status")
            st.json(gpu_status)
        else:
            st.error("GPU status not available")

    def show_quantum_ai(self):
        """Show quantum AI interface"""
        st.header("Quantum AI")

        if not self.revenue_optimizer:
            st.error("Revenue optimizer not available")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Quantum Portfolio Optimization")
            if st.button("Run Quantum Portfolio Optimization"):
                try:
                    result = self.revenue_optimizer.optimize_quantum_portfolio()
                    st.success("Quantum portfolio optimization completed!")
                    st.json(result.__dict__)
                except Exception as e:
                    st.error(f"Quantum optimization failed: {e}")

        with col2:
            st.subheader("Quantum Risk Analysis")
            if st.button("Run Quantum Risk Analysis"):
                try:
                    result = self.revenue_optimizer.analyze_quantum_risk()
                    st.success("Quantum risk analysis completed!")
                    st.json(result.__dict__)
                except Exception as e:
                    st.error(f"Quantum risk analysis failed: {e}")

        # Market prediction
        st.subheader("Quantum Market Prediction")
        symbol = st.text_input("Stock Symbol", "TECH_STOCK")
        if st.button("Predict Market Movement"):
            try:
                prediction = self.revenue_optimizer.predict_market_with_quantum(symbol)
                st.success("Quantum market prediction completed!")
                st.json(prediction.__dict__)
            except Exception as e:
                st.error(f"Market prediction failed: {e}")

    def show_database(self):
        """Show database interface"""
        st.header("Database Management")

        if not self.db_manager:
            st.error("Database manager not available")
            return

        # Database status
        status = self.db_manager.get_database_status()
        st.subheader("Database Status")
        st.json(status)

        # Recent predictions
        st.subheader("Recent Predictions")
        predictions = self.db_manager.get_predictions(limit=20)
        if predictions:
            pred_df = pd.DataFrame(predictions)
            st.dataframe(pred_df)

    def show_system_health(self):
        """Show system health dashboard"""
        st.header("System Health")

        status = self.get_system_status()
        if status:
            st.subheader("Service Status")

            for service, active in status.get('services', {}).items():
                status_class = "status-good" if active else "status-error"
                st.markdown(f'<span class="{status_class}">{service}: {"Active" if active else "Inactive"}</span>', unsafe_allow_html=True)

        # Performance metrics
        if self.db_manager:
            st.subheader("System Metrics")
            # This would show metrics from the database
            st.info("Database metrics visualization would be implemented here")

    def show_settings(self):
        """Show settings page"""
        st.header("Settings")

        st.subheader("API Configuration")
        api_url = st.text_input("API Base URL", API_BASE_URL)
        if st.button("Update API URL"):
            self.api_url = api_url
            st.success("API URL updated!")

        st.subheader("Database Configuration")
        if self.db_manager:
            db_status = self.db_manager.get_database_status()
            st.json(db_status)

    def get_system_status(self) -> Optional[Dict]:
        """Get system status from API"""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Failed to get system status: {e}")
        return None

    def get_gpu_status(self) -> Optional[Dict]:
        """Get GPU status from API"""
        try:
            response = requests.get(f"{self.api_url}/gpu/status", timeout=5)
            if response.status_code == 200:
                return response.json().get('gpu_status', {})
        except Exception as e:
            st.error(f"Failed to get GPU status: {e}")
        return None

def main():
    """Main dashboard function"""
    dashboard = AIDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
