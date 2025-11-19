"""
Swagger/OpenAPI Configuration
Provides comprehensive API documentation
"""
from flask_restx import Api, Resource, fields, Namespace
from flask import Blueprint

# Create API blueprint
api_blueprint = Blueprint('api', __name__, url_prefix='/api')

# Initialize Flask-RESTX API
api = Api(
    api_blueprint,
    version='1.0.0',
    title='JPMorgan Financial APIs',
    description='Enterprise-grade API service for financial data processing and management',
    doc='/docs/',
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'Add "Bearer " prefix to your token'
        }
    },
    security='Bearer'
)

# Create namespaces
ns_auth = Namespace('auth', description='Authentication operations')
ns_telemetry = Namespace('telemetry', description='Telemetry data operations')
ns_business = Namespace('business', description='Business management operations')
ns_assets = Namespace('assets', description='Asset management operations')
ns_ml = Namespace('ml', description='Machine learning operations')
ns_jpmorgan = Namespace('jpmorgan', description='JPMorgan Private Bank services')

# Add namespaces to API
api.add_namespace(ns_auth, path='/auth')
api.add_namespace(ns_telemetry, path='/telemetry')
api.add_namespace(ns_business, path='/businesses')
api.add_namespace(ns_assets, path='/assets')
api.add_namespace(ns_ml, path='/ml')
api.add_namespace(ns_jpmorgan, path='/jpmorgan')

# Define models for request/response documentation

# Authentication models
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username', example='testuser'),
    'password': fields.String(required=True, description='Password', example='password123')
})

register_model = api.model('Register', {
    'username': fields.String(required=True, description='Username', example='newuser'),
    'password': fields.String(required=True, description='Password (min 8 chars, uppercase, lowercase, digit)', example='SecurePass123'),
    'email': fields.String(required=True, description='Email address', example='user@example.com')
})

token_response = api.model('TokenResponse', {
    'token': fields.String(description='Authentication token'),
    'user': fields.String(description='Username'),
    'created_at': fields.String(description='Token creation timestamp')
})

# Business models
business_model = api.model('Business', {
    'name': fields.String(required=True, description='Business name', example='Acme Corporation'),
    'type': fields.String(required=True, description='Business type', enum=['corporation', 'llc', 'partnership', 'sole_proprietorship', 'nonprofit'], example='corporation'),
    'registration_number': fields.String(required=True, description='Registration number', example='123456789'),
    'address': fields.String(description='Business address', example='123 Main St, New York, NY 10001'),
    'contact_info': fields.Raw(description='Contact information', example={'email': 'contact@acme.com', 'phone': '+1-555-0123'})
})

business_response = api.model('BusinessResponse', {
    'id': fields.Integer(description='Business ID'),
    'name': fields.String(description='Business name'),
    'type': fields.String(description='Business type'),
    'registration_number': fields.String(description='Registration number'),
    'address': fields.String(description='Business address'),
    'contact_info': fields.Raw(description='Contact information'),
    'created_at': fields.String(description='Creation timestamp')
})

# Asset models
asset_model = api.model('Asset', {
    'business_id': fields.Integer(required=True, description='Business ID', example=1),
    'name': fields.String(required=True, description='Asset name', example='Office Building'),
    'type': fields.String(required=True, description='Asset type', enum=['equipment', 'property', 'vehicle', 'intellectual_property', 'other'], example='property'),
    'value': fields.Float(required=True, description='Asset value', example=500000.00),
    'acquisition_date': fields.String(description='Acquisition date (ISO 8601)', example='2023-01-15T00:00:00Z'),
    'ownership_percentage': fields.Float(description='Ownership percentage', example=100.0),
    'description': fields.String(description='Asset description', example='Commercial office building')
})

asset_response = api.model('AssetResponse', {
    'id': fields.Integer(description='Asset ID'),
    'business_id': fields.Integer(description='Business ID'),
    'name': fields.String(description='Asset name'),
    'type': fields.String(description='Asset type'),
    'value': fields.Float(description='Asset value'),
    'acquisition_date': fields.String(description='Acquisition date'),
    'ownership_percentage': fields.Float(description='Ownership percentage'),
    'description': fields.String(description='Asset description'),
    'created_at': fields.String(description='Creation timestamp')
})

# Telemetry models
telemetry_model = api.model('Telemetry', {
    'ver': fields.String(required=True, description='Version', example='4.0'),
    'name': fields.String(required=True, description='Event name', example='Microsoft.Windows.Event'),
    'time': fields.String(required=True, description='Event timestamp (ISO 8601)', example='2023-01-15T12:00:00Z'),
    'data': fields.Raw(description='Event data', example={'key': 'value'})
})

telemetry_batch_model = api.model('TelemetryBatch', {
    'events': fields.List(fields.Nested(telemetry_model), required=True, description='List of telemetry events')
})

# ML models
ml_train_model = api.model('MLTrain', {
    'training_data': fields.List(fields.Raw, required=True, description='Training data'),
    'contamination': fields.Float(description='Contamination factor', example=0.1)
})

ml_anomaly_model = api.model('MLAnomaly', {
    'data': fields.List(fields.Raw, required=True, description='Data to analyze for anomalies')
})

# Error response model
error_response = api.model('ErrorResponse', {
    'status': fields.String(description='Status', example='error'),
    'error': fields.String(description='Error message'),
    'error_code': fields.String(description='Error code'),
    'timestamp': fields.String(description='Timestamp')
})

# Success response model
success_response = api.model('SuccessResponse', {
    'status': fields.String(description='Status', example='success'),
    'message': fields.String(description='Success message'),
    'timestamp': fields.String(description='Timestamp')
})

# Health check model
health_response = api.model('HealthResponse', {
    'status': fields.String(description='Health status', example='healthy'),
    'timestamp': fields.String(description='Timestamp'),
    'version': fields.String(description='API version'),
    'uptime': fields.Float(description='Uptime in seconds')
})

# Metrics model
metrics_response = api.model('MetricsResponse', {
    'total_events': fields.Integer(description='Total events processed'),
    'events_per_hour': fields.Float(description='Events per hour'),
    'average_processing_time': fields.Float(description='Average processing time (ms)'),
    'error_rate': fields.Float(description='Error rate percentage'),
    'timestamp': fields.String(description='Timestamp')
})

def configure_swagger(app):
    """
    Configure Swagger documentation for the Flask app

    Args:
        app: Flask application instance
    """
    # Register API blueprint
    app.register_blueprint(api_blueprint)

    # Add custom CSS for Swagger UI
    @api.documentation
    def custom_ui():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>JPMorgan Financial APIs - Documentation</title>
            <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.min.css">
            <style>
                .swagger-ui .topbar { background-color: #003366; }
                .swagger-ui .info .title { color: #003366; }
            </style>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-standalone-preset.min.js"></script>
            <script>
                window.onload = function() {
                    SwaggerUIBundle({
                        url: '/api/swagger.json',
                        dom_id: '#swagger-ui',
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIStandalonePreset
                        ],
                        layout: "StandaloneLayout"
                    });
                };
            </script>
        </body>
        </html>
        '''

    return api

# Example endpoint documentation
@ns_auth.route('/login')
class Login(Resource):
    @ns_auth.doc('login')
    @ns_auth.expect(login_model)
    @ns_auth.marshal_with(token_response, code=200)
    @ns_auth.response(401, 'Invalid credentials', error_response)
    def post(self):
        """Authenticate user and return token"""
        pass

@ns_auth.route('/register')
class Register(Resource):
    @ns_auth.doc('register')
    @ns_auth.expect(register_model)
    @ns_auth.marshal_with(success_response, code=201)
    @ns_auth.response(400, 'Validation error', error_response)
    def post(self):
        """Register new user"""
        pass

@ns_business.route('/')
class BusinessList(Resource):
    @ns_business.doc('list_businesses', security='Bearer')
    @ns_business.marshal_list_with(business_response)
    @ns_business.response(401, 'Unauthorized', error_response)
    def get(self):
        """List all businesses"""
        pass

    @ns_business.doc('create_business', security='Bearer')
    @ns_business.expect(business_model)
    @ns_business.marshal_with(business_response, code=201)
    @ns_business.response(400, 'Validation error', error_response)
    @ns_business.response(401, 'Unauthorized', error_response)
    def post(self):
        """Create new business"""
        pass

@ns_business.route('/<int:id>')
class Business(Resource):
    @ns_business.doc('get_business', security='Bearer')
    @ns_business.marshal_with(business_response)
    @ns_business.response(404, 'Business not found', error_response)
    @ns_business.response(401, 'Unauthorized', error_response)
    def get(self, id):
        """Get business by ID"""
        pass

    @ns_business.doc('update_business', security='Bearer')
    @ns_business.expect(business_model)
    @ns_business.marshal_with(business_response)
    @ns_business.response(404, 'Business not found', error_response)
    @ns_business.response(401, 'Unauthorized', error_response)
    def put(self, id):
        """Update business"""
        pass

    @ns_business.doc('delete_business', security='Bearer')
    @ns_business.marshal_with(success_response)
    @ns_business.response(404, 'Business not found', error_response)
    @ns_business.response(401, 'Unauthorized', error_response)
    def delete(self, id):
        """Delete business"""
        pass

@ns_telemetry.route('/')
class TelemetryEvent(Resource):
    @ns_telemetry.doc('post_telemetry', security='Bearer')
    @ns_telemetry.expect(telemetry_model)
    @ns_telemetry.marshal_with(success_response, code=201)
    @ns_telemetry.response(400, 'Validation error', error_response)
    @ns_telemetry.response(401, 'Unauthorized', error_response)
    def post(self):
        """Submit telemetry event"""
        pass

@ns_telemetry.route('/batch')
class TelemetryBatch(Resource):
    @ns_telemetry.doc('post_telemetry_batch', security='Bearer')
    @ns_telemetry.expect(telemetry_batch_model)
    @ns_telemetry.marshal_with(success_response, code=201)
    @ns_telemetry.response(400, 'Validation error', error_response)
    @ns_telemetry.response(401, 'Unauthorized', error_response)
    def post(self):
        """Submit batch of telemetry events"""
        pass

@ns_telemetry.route('/metrics')
class TelemetryMetrics(Resource):
    @ns_telemetry.doc('get_metrics')
    @ns_telemetry.marshal_with(metrics_response)
    def get(self):
        """Get telemetry metrics"""
        pass
