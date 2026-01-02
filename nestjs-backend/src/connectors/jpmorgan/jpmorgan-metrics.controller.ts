import { Controller, Get, Header } from '@nestjs/common';
import { JpmorganMetricsService } from './jpmorgan-metrics.service';

@Controller('metrics')
export class JpmorganMetricsController {
  constructor(private readonly metricsService: JpmorganMetricsService) {}

  @Get()
  @Header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
  async getMetrics(): Promise<string> {
    return this.metricsService.getMetrics();
  }
}
