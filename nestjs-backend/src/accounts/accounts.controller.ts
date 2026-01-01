import { Controller, Get, Param, Post } from '@nestjs/common';
import { AccountsService } from './accounts.service';

@Controller('accounts')
export class AccountsController {
  constructor(private readonly accountsService: AccountsService) {}

  @Get('organization/:orgId')
  async listForOrg(@Param('orgId') orgId: string) {
    return this.accountsService.listAccountsByOrganization(orgId);
  }

  @Post('sync/connection/:connectionId')
  async sync(@Param('connectionId') connectionId: string) {
    return this.accountsService.syncAccountsForConnection(connectionId);
  }
}
