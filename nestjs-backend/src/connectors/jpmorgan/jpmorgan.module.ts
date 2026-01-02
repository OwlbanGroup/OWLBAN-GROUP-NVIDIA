import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { JpmorganService } from './jpmorgan.service';
import { JpmorganTokenService } from './jpmorgan-token.service';
import { JpmorganController } from './jpmorgan.controller';
import { JpmorganMetricsService } from './jpmorgan-metrics.service';
import { JpmorganMetricsController } from './jpmorgan-metrics.controller';

@Module({
  imports: [
    HttpModule.register({
      timeout: 30000, // 30 seconds for JPMorgan API calls
      maxRedirects: 5,
    }),
  ],
  controllers: [JpmorganController, JpmorganMetricsController],
  providers: [JpmorganMetricsService, JpmorganTokenService, JpmorganService],
  exports: [JpmorganService, JpmorganTokenService, JpmorganMetricsService],
})
export class JpmorganModule {}
