import { Controller, Get } from '@nestjs/common';
import {
  HealthCheck,
  HealthCheckService,
  TypeOrmHealthIndicator,
  MemoryHealthIndicator,
  DiskHealthIndicator,
} from '@nestjs/terminus';
import { ConfigService } from '@nestjs/config';

@Controller('health')
export class HealthController {
  constructor(
    private health: HealthCheckService,
    private db: TypeOrmHealthIndicator,
    private memory: MemoryHealthIndicator,
    private disk: DiskHealthIndicator,
    private config: ConfigService,
  ) {}

  @Get()
  @HealthCheck()
  check() {
    return this.health.check([
      // Database health check
      () => this.db.pingCheck('database'),
      
      // Memory health check (heap should not exceed 150MB)
      () => this.memory.checkHeap('memory_heap', 150 * 1024 * 1024),
      
      // Memory health check (RSS should not exceed 300MB)
      () => this.memory.checkRSS('memory_rss', 300 * 1024 * 1024),
      
      // Disk health check (storage should not exceed 90% of available space)
      () =>
        this.disk.checkStorage('storage', {
          path: '/',
          thresholdPercent: 0.9,
        }),
    ]);
  }

  @Get('liveness')
  @HealthCheck()
  liveness() {
    return this.health.check([
      // Simple check to verify the application is running
      () => ({ liveness: { status: 'up' } }),
    ]);
  }

  @Get('readiness')
  @HealthCheck()
  readiness() {
    return this.health.check([
      // Check if database is ready
      () => this.db.pingCheck('database'),
    ]);
  }

  @Get('/api/system/status')
  async getSystemStatus() {
    const healthChecks = await this.health.check([
      () => this.db.pingCheck('database'),
      () => this.memory.checkHeap('memory_heap', 150 * 1024 * 1024),
      () => this.memory.checkRSS('memory_rss', 300 * 1024 * 1024),
    ]);

    return {
      status: 'operational',
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      environment: this.config.get('NODE_ENV') || 'development',
      services: {
        api: {
          status: 'up',
          uptime: process.uptime(),
        },
        database: {
          status: healthChecks.details?.database?.status || 'unknown',
        },
        jpmorgan: {
          status: 'up',
          baseUrl: this.config.get('JPM_API_BASE_URL') || 'https://api-sandbox.payments.jpmorgan.com',
        },
      },
      system: {
        memory: {
          heap: healthChecks.details?.memory_heap || {},
          rss: healthChecks.details?.memory_rss || {},
        },
        nodeVersion: process.version,
        platform: process.platform,
      },
    };
  }
}
