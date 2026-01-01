import { registerAs } from '@nestjs/config';
import { TypeOrmModuleOptions } from '@nestjs/typeorm';

export default registerAs(
  'database',
  (): TypeOrmModuleOptions => ({
    type: 'postgres',
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT || '5432', 10),
    username: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    autoLoadEntities: true,
    synchronize: false, // Always false in production
    logging: process.env.NODE_ENV === 'development' ? ['query', 'error'] : ['error'],
    
    // Connection Pool Configuration
    extra: {
      max: parseInt(process.env.DB_POOL_SIZE || '10', 10),
      min: 2,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: parseInt(process.env.DB_CONNECTION_TIMEOUT || '30000', 10),
    },

    // Retry Logic
    retryAttempts: 3,
    retryDelay: 3000,

    // SSL Configuration for production
    ssl: process.env.NODE_ENV === 'production' ? {
      rejectUnauthorized: false,
    } : false,

    // Migration Configuration
    migrations: ['dist/migrations/**/*.js'],
    migrationsRun: false,
    migrationsTableName: 'migrations',
  }),
);
