import { Controller, Get, Query } from '@nestjs/common';
import { FinancialService } from './financial.service';
import { FinancialSummaryDto } from './dtos/financial-summary.dto';
import { AssetsResponseDto } from './dtos/assets-response.dto';
import { PerformanceResponseDto } from './dtos/performance-response.dto';
import { StocksResponseDto } from './dtos/stocks-response.dto';

@Controller('api/financial')
export class FinancialController {
  constructor(private readonly financialService: FinancialService) {}

  @Get('summary')
  async getFinancialSummary(
    @Query('orgId') orgId?: string,
  ): Promise<FinancialSummaryDto> {
    return this.financialService.getFinancialSummary(orgId);
  }

  @Get('assets')
  async getAssets(@Query('orgId') orgId?: string): Promise<AssetsResponseDto> {
    return this.financialService.getAssets(orgId);
  }

  @Get('performance')
  async getPerformance(
    @Query('orgId') orgId?: string,
  ): Promise<PerformanceResponseDto> {
    return this.financialService.getPerformance(orgId);
  }

  @Get('stocks')
  async getStocks(@Query('orgId') orgId?: string): Promise<StocksResponseDto> {
    return this.financialService.getStocks(orgId);
  }
}
