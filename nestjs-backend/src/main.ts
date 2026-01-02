import { NestFactory } from '@nestjs/core';
import { ValidationPipe, VersioningType } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import helmet from 'helmet';
import compression from 'compression';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    logger: ['error', 'warn', 'log', 'debug', 'verbose'],
  });

  const configService = app.get(ConfigService);

  // Security Middleware
  app.use(helmet());
  app.use(compression());

  // CORS Configuration
  app.enableCors({
    origin: configService.get('CORS_ORIGIN') || '*',
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  });

  // Global Validation Pipe
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true, // Strip properties that don't have decorators
      forbidNonWhitelisted: true, // Throw error if non-whitelisted properties are present
      transform: true, // Automatically transform payloads to DTO instances
      transformOptions: {
        enableImplicitConversion: true,
      },
      disableErrorMessages: configService.get('NODE_ENV') === 'production',
    }),
  );

  // API Versioning
  app.enableVersioning({
    type: VersioningType.URI,
    defaultVersion: '1',
    prefix: 'api/v',
  });

  // Global Prefix
  const apiPrefix = configService.get('API_PREFIX') || 'api';
  app.setGlobalPrefix(apiPrefix);

  // Swagger Documentation
  if (configService.get('NODE_ENV') !== 'production') {
    const config = new DocumentBuilder()
      .setTitle('JPMorgan Financial APIs')
      .setDescription('Comprehensive financial services API')
      .setVersion('1.0')
      .addBearerAuth()
      .addTag('auth', 'Authentication endpoints')
      .addTag('users', 'User management')
      .addTag('organizations', 'Organization management')
      .addTag('bank-connections', 'Bank connection management')
      .addTag('accounts', 'Account management')
      .addTag('balances', 'Balance information')
      .addTag('transactions', 'Transaction management')
      .addTag('payments', 'Payment processing')
      .addTag('payroll', 'Payroll management')
      .addTag('petty-cash', 'Petty cash management')
      .addTag('corporate', 'Corporate services')
      .addTag('jpmorgan', 'JPMorgan connector')
      .build();

    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api/docs', app, document, {
      swaggerOptions: {
        persistAuthorization: true,
      },
    });
  }

  // Graceful Shutdown
  app.enableShutdownHooks();

  const port = configService.get('PORT') || 3000;
  await app.listen(port);

  console.log(`
    🚀 Application is running on: http://localhost:${port}/${apiPrefix}
    📚 API Documentation: http://localhost:${port}/api/docs
    🏥 Health Check: http://localhost:${port}/health
    🔒 Environment: ${configService.get('NODE_ENV')}
  `);
}

bootstrap().catch((error) => {
  console.error('Failed to start application:', error);
  process.exit(1);
});
