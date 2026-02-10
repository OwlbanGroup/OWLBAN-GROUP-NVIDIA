"""
AI Service module for JPMorgan Financial APIs using LangChain and LangSmith
Provides AI-powered financial analysis, insights, and natural language processing capabilities
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langsmith import Client as LangSmithClient

from config import config
from src.logger import telemetry_logger

class AIService:
    """AI Service for financial data analysis and insights using LangChain"""

    def __init__(self):
        """Initialize AI service with LangChain and LangSmith"""
        self.logger = telemetry_logger.get_logger()

        # Initialize LangSmith client for tracing
        if config.LANGCHAIN_API_KEY:
            self.langsmith_client = LangSmithClient(
                api_key=config.LANGCHAIN_API_KEY,
                api_url=config.LANGCHAIN_ENDPOINT
            )
        else:
            self.langsmith_client = None
            self.logger.warning("LangSmith API key not configured - tracing disabled")

        # Initialize LLM - prefer Blackbox AI if configured, fallback to OpenAI
        self.llm = None
        self.llm_provider = None

        # Try Blackbox AI first
        if config.BLACKBOX_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=config.BLACKBOX_MODEL,
                    temperature=config.BLACKBOX_TEMPERATURE,
                    openai_api_key=config.BLACKBOX_API_KEY,
                    openai_api_base=config.BLACKBOX_BASE_URL
                )
                self.llm_provider = "blackbox"
                self.logger.info("Blackbox AI initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Blackbox AI: {str(e)}")

        # Fallback to OpenAI if Blackbox not available or failed
        if not self.llm and config.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=config.OPENAI_MODEL,
                    temperature=config.OPENAI_TEMPERATURE,
                    openai_api_key=config.OPENAI_API_KEY,
                    callbacks=[LangChainTracer(project_name=config.LANGCHAIN_PROJECT)] if self.langsmith_client else []
                )
                self.llm_provider = "openai"
                self.logger.info("OpenAI initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI: {str(e)}")

        if not self.llm:
            self.logger.warning("No AI provider configured - AI services disabled")

        # Initialize prompt templates
        self._setup_prompts()

        self.logger.info("AI Service initialized successfully")

    def _setup_prompts(self):
        """Set up prompt templates for different AI tasks"""

        # Financial analysis prompt
        self.financial_analysis_prompt = PromptTemplate(
            input_variables=["data", "context", "question"],
            template="""
            You are a senior financial analyst at JPMorgan Chase. Analyze the following financial data and provide insights.

            Context: {context}
            Data: {data}

            Question: {question}

            Provide a detailed analysis including:
            1. Key trends and patterns
            2. Risk assessment
            3. Recommendations
            4. Potential impact on business decisions

            Be specific, data-driven, and professional in your analysis.
            """
        )

        # Risk assessment prompt
        self.risk_assessment_prompt = PromptTemplate(
            input_variables=["transaction_data", "historical_patterns", "market_conditions"],
            template="""
            You are a risk management expert at JPMorgan Chase. Assess the risk profile of the following transaction data.

            Transaction Data: {transaction_data}
            Historical Patterns: {historical_patterns}
            Market Conditions: {market_conditions}

            Provide a comprehensive risk assessment including:
            1. Risk level (Low/Medium/High/Critical)
            2. Key risk factors identified
            3. Mitigation strategies
            4. Recommended actions
            5. Confidence level in assessment

            Base your assessment on industry best practices and regulatory requirements.
            """
        )

        # Natural language query prompt
        self.nl_query_prompt = PromptTemplate(
            input_variables=["query", "data_schema", "available_data"],
            template="""
            You are a financial data assistant for JPMorgan Chase APIs. Help users query and understand financial data.

            User Query: {query}

            Available Data Schema: {data_schema}
            Current Data Context: {available_data}

            Provide:
            1. Interpretation of the user's query
            2. Relevant data insights
            3. Suggested API calls or data analysis
            4. Clear, actionable response

            If the query cannot be answered with available data, suggest what additional information would be needed.
            """
        )

    def analyze_financial_data(self, data: Dict[str, Any], question: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze financial data using AI

        Args:
            data: Financial data to analyze
            question: Specific question about the data
            context: Additional context for analysis

        Returns:
            Analysis results with insights and recommendations
        """
        try:
            chain = self.financial_analysis_prompt | self.llm

            result = chain.invoke({
                "data": str(data),
                "context": context,
                "question": question
            })

            model_used = config.BLACKBOX_MODEL if self.llm_provider == "blackbox" else config.OPENAI_MODEL
            return {
                "status": "success",
                "analysis": result.content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": model_used
            }

        except Exception as e:
            self.logger.error(f"Error in financial data analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def assess_transaction_risk(self, transaction_data: Dict[str, Any],
                                    historical_patterns: List[Dict] = None,
                                    market_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess risk of financial transactions using AI

        Args:
            transaction_data: Transaction details to assess
            historical_patterns: Historical transaction patterns
            market_conditions: Current market conditions

        Returns:
            Risk assessment with recommendations
        """
        try:
            chain = self.risk_assessment_prompt | self.llm

            result = chain.invoke({
                "transaction_data": str(transaction_data),
                "historical_patterns": str(historical_patterns or []),
                "market_conditions": str(market_conditions or {})
            })

            model_used = config.BLACKBOX_MODEL if self.llm_provider == "blackbox" else config.OPENAI_MODEL
            return {
                "status": "success",
                "risk_assessment": result.content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": model_used
            }

        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def process_natural_language_query(self, query: str,
                                           data_schema: Dict[str, Any],
                                           available_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process natural language queries about financial data

        Args:
            query: User's natural language query
            data_schema: Schema of available data
            available_data: Current data context

        Returns:
            Processed query response with insights
        """
        try:
            chain = self.nl_query_prompt | self.llm

            result = chain.invoke({
                "query": query,
                "data_schema": str(data_schema),
                "available_data": str(available_data or {})
            })

            model_used = config.BLACKBOX_MODEL if self.llm_provider == "blackbox" else config.OPENAI_MODEL
            return {
                "status": "success",
                "response": result.content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": model_used
            }

        except Exception as e:
            self.logger.error(f"Error in natural language query processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_service_status(self) -> Dict[str, Any]:
        """Get AI service status and configuration"""
        model = config.BLACKBOX_MODEL if self.llm_provider == "blackbox" else config.OPENAI_MODEL
        temperature = config.BLACKBOX_TEMPERATURE if self.llm_provider == "blackbox" else config.OPENAI_TEMPERATURE

        return {
            "status": "healthy" if self.llm else "unhealthy",
            "provider": self.llm_provider or "none",
            "model": model,
            "temperature": temperature,
            "langsmith_enabled": self.langsmith_client is not None,
            "project": config.LANGCHAIN_PROJECT,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global AI service instance
ai_service = AIService()
