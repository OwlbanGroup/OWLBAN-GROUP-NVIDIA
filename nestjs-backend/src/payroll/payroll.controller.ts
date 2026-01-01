import { Controller, Post, Get, Param, Body } from '@nestjs/common';
import { PayrollService } from './payroll.service';

@Controller('payroll')
export class PayrollController {
  constructor(private readonly payroll: PayrollService) {}

  @Post('employee/:orgId')
  addEmployee(@Param('orgId') orgId: string, @Body() dto: any) {
    return this.payroll.addEmployee(orgId, dto);
  }

  @Get('employees/:orgId')
  listEmployees(@Param('orgId') orgId: string) {
    return this.payroll.listEmployees(orgId);
  }

  @Post('run/:orgId')
  createRun(@Param('orgId') orgId: string, @Body() dto: any) {
    return this.payroll.createPayrollRun(orgId, dto);
  }

  @Post('execute/:runId')
  executeRun(
    @Param('runId') runId: string,
    @Body('debitAccountId') debitAccountId: string,
  ) {
    return this.payroll.executePayrollRun(runId, debitAccountId);
  }

  @Get('runs/:orgId')
  listRuns(@Param('orgId') orgId: string) {
    return this.payroll.listPayrollRuns(orgId);
  }

  @Get('run/:runId')
  getRun(@Param('runId') runId: string) {
    return this.payroll.getRunWithPayments(runId);
  }
}
